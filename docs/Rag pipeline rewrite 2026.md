# RAG Pipeline — Full Rewrite Spec (2026)
**Target: Tamil + English + Tanglish, query-language-matched answers, source-cited, CPU-now/GPU-later, self-hosted Mongo + Qdrant on Kubernetes**

---

## 1. Locked-in decisions

| Area | Decision | Reasoning |
|---|---|---|
| Vector store | **Qdrant**, k8s service, internal `http://qdrant:6333` | Already in your env config; self-hosted Mongo has no `$vectorSearch` |
| Document/text store | **Mongo stays** (`pdf_ocr` db) — source of truth for chunk text + metadata | No need to migrate; Qdrant only stores vectors + filter fields |
| Embedding model (CPU, now) | **`intfloat/multilingual-e5-base`** (768-dim) | CPU-feasible at 10k–100k chunks, strong Tamil/English/cross-lingual |
| Embedding model (GPU, later) | **`BAAI/bge-m3`** (1024-dim) | Matches your `QDRANT_VECTOR_SIZE=1024`; native hybrid dense+sparse |
| LLM (now) | **Mistral-7B**, kept, prompt + language-detection fixed | "Use available option now" |
| LLM (upgrade path) | `sarvam-m` or `ai4bharat/Airavata` (swap via env var, no code change) | Stronger Tamil generation when ready |
| Switch already in your hands | `USE_GPU=false` → drives model selection automatically | One flag controls embed model + Qdrant dim |

**Important constraint to flag clearly:** the `768` (CPU model) vs `1024` (GPU model) dimension mismatch means **switching `USE_GPU` requires re-creating the Qdrant collection and re-embedding everything** — it is not a hot-swap. Section 7 has the migration runbook for this.

---

## 2. New end-to-end architecture

```
📄 PDF Upload
   ↓
OCR Engine (unchanged — PaddleVL / DocAI / Surya)
   ↓
core/worker.py → pages/page_NNNN.json (unchanged)
   ↓
core/chunking.py  [REWRITTEN]
   • heading-aware chunking (carries last-seen heading forward)
   • cross-page overlap (bridges page boundaries)
   • per-chunk language tag (ta / en / mixed)
   • enriched metadata (doc title, parent heading, position, language)
   ↓
core/embeddings.py  [REWRITTEN]
   • model selected by USE_GPU flag
   • writes to BOTH Mongo (text+meta) AND Qdrant (vector+filters)
   ↓
MongoDB: chunks collection (text, full metadata, chunk_id)
Qdrant: documents collection (vector, chunk_id, doc_id, language, page — filter-only fields)
   ↓
🔍 User Query (Tamil / English / Tanglish)
   ↓
core/retrieval_pipeline.py  [NEW — shared by API + Voice]
   1. QueryProcessor: language detection + filler removal + variant generation
   2. Multi-query embedding (same model as corpus)
   3. Qdrant search → top-200 candidates (replaces "load all chunks")
   4. Hydrate candidate text from Mongo by chunk_id (only 200, not 5000+)
   5. Stage 1: Hybrid score (Qdrant cosine + BM25 in-process + section boost)
   6. Stage 2: RRF re-rank
   7. Stage 3 [NEW]: MMR dedup
   8. Relevance gating (MIN_CONTEXT_SCORE=0.35, unchanged)
   ↓
LLM Answer Generation
   • detected query language passed into prompt
   • "answer in the same language as the question" instruction
   • citations carry section_label + page + doc title
   ↓
📝 Answer (Tamil-in/Tamil-out, English-in/English-out, Tanglish-in/best-effort match) + Citations
```

Both `api/semantic_search.py` and `voice/conversation_engine.py` call **the same** `retrieval_pipeline.py` — this is the Issue 5 fix, and it's structural now, not a convention to remember.

---

## 3. Component-by-component changes

### 3.1 `core/chunking.py` (replaces relevant parts of `core/semantic_search.py`)

**Heading-aware chunking** — fixes Issue 1:

```python
def build_search_chunks(structured_elements, doc_metadata):
    chunks = []
    current_heading = None
    current_heading_level = None

    for element in structured_elements:
        if element.type in ("heading", "title", "section_header"):
            current_heading = element.text
            current_heading_level = element.level
            continue  # headings become context, not standalone chunks unless orphaned

        # paragraph / list / table element
        chunk_text = element.text
        if current_heading:
            # prefix becomes part of embedded text AND stored separately for citation
            embed_text = f"{current_heading}\n\n{chunk_text}"
        else:
            embed_text = chunk_text

        chunks.append(make_chunk(
            text=chunk_text,
            embed_text=embed_text,
            parent_heading=current_heading,
            heading_level=current_heading_level,
            element=element,
            doc_metadata=doc_metadata,
        ))

    return chunks
```

Key point: **store `text` (clean, for display/citation) separately from `embed_text` (heading-prefixed, for embedding only)**. Otherwise every citation shown to the user repeats the heading awkwardly.

**Cross-page overlap** — fixes Issue 4:

```python
def add_cross_page_bridges(pages_text, overlap_chars=150):
    """After normal per-page chunking, add bridge chunks spanning page boundaries."""
    bridge_chunks = []
    for i in range(len(pages_text) - 1):
        tail = pages_text[i][-overlap_chars:]
        head = pages_text[i + 1][:overlap_chars]
        bridge_text = tail + " " + head
        bridge_chunks.append(make_chunk(
            text=bridge_text,
            embed_text=bridge_text,
            is_bridge=True,
            spans_pages=(i + 1, i + 2),  # 1-indexed page numbers
        ))
    return bridge_chunks
```

Bridge chunks are tagged `is_bridge: true` so they can be down-weighted slightly in scoring if you don't want them dominating (they're often redundant with their neighbors, but they're cheap insurance against split sentences).

**Enriched metadata** — fixes Issue 6:

```python
def make_chunk(text, embed_text, doc_metadata, parent_heading=None,
                heading_level=None, element=None, is_bridge=False, spans_pages=None):
    return {
        "chunk_id": str(uuid4()),
        "text": text,
        "embed_text": embed_text,
        "doc_id": doc_metadata["doc_id"],
        "doc_title": doc_metadata["title"],
        "filename": doc_metadata["filename"],
        "page": element.page if element else None,
        "element_type": element.type if element else "bridge",
        "parent_heading": parent_heading,
        "heading_level": heading_level,
        "section_label": parent_heading or doc_metadata.get("default_section"),
        "position_in_doc": element.index / doc_metadata["total_elements"] if element else None,  # 0.0–1.0
        "language": detect_language(text),   # "ta" | "en" | "mixed"
        "is_bridge": is_bridge,
        "spans_pages": spans_pages,
    }
```

`detect_language()` — cheap, no model needed:

```python
import re

TAMIL_RANGE = re.compile(r'[\u0B80-\u0BFF]')

def detect_language(text: str) -> str:
    tamil_chars = len(TAMIL_RANGE.findall(text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        return "en"
    tamil_ratio = tamil_chars / total_chars
    if tamil_ratio > 0.4:
        return "ta"
    elif tamil_ratio > 0.05:
        return "mixed"   # Tanglish or code-mixed
    return "en"
```

This only detects Tamil **script**. Tanglish written in Latin script (e.g. "stamp duty eppadi calculate pannanum") will register as `"en"` by character set — that's fine, because language *detection for retrieval purposes* matters less than language detection *for the answer*, which happens at the LLM step using the actual query (Section 5).

---

### 3.2 `core/embeddings.py` — model selection by `USE_GPU`

```python
import os
from sentence_transformers import SentenceTransformer

USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

if USE_GPU:
    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024
    DEVICE = "cuda"
else:
    MODEL_NAME = "intfloat/multilingual-e5-base"
    EMBEDDING_DIM = 768
    DEVICE = "cpu"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model

def embed_text(text: str, is_query: bool = False) -> list[float]:
    model = get_model()
    # e5 models require a prefix convention — bge-m3 does not need this,
    # so this is conditional on which model is active
    if "e5" in MODEL_NAME:
        prefixed = f"query: {text}" if is_query else f"passage: {text}"
        return model.encode(prefixed, normalize_embeddings=True).tolist()
    return model.encode(text, normalize_embeddings=True).tolist()

def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = get_model()
    if "e5" in MODEL_NAME:
        texts = [f"query: {t}" if is_query else f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
```

**`e5` models need the `query:` / `passage:` prefix convention to perform correctly** — this is a known easy-to-miss detail when switching to e5-family models; without it, retrieval quality silently degrades.

`.env` additions:

```bash
# Embedding model selection — auto-switches with USE_GPU
USE_GPU=false
EMBED_MODEL_CPU=intfloat/multilingual-e5-base
EMBED_MODEL_GPU=BAAI/bge-m3
EMBED_DIM_CPU=768
EMBED_DIM_GPU=1024
```

---

### 3.3 Qdrant integration — fixes Issue 2

```python
# db/qdrant_client.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
VECTOR_SIZE = 1024 if USE_GPU else 768

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
    prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
)

def ensure_collection():
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def upsert_chunk(chunk_id: str, vector: list[float], filter_fields: dict):
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(
            id=chunk_id,
            vector=vector,
            payload={
                "doc_id": filter_fields["doc_id"],
                "language": filter_fields["language"],
                "page": filter_fields["page"],
                "is_bridge": filter_fields.get("is_bridge", False),
            },
        )],
    )

def search(query_vector: list[float], top_k: int = 200, doc_id_filter: str = None):
    query_filter = None
    if doc_id_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))])

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [(r.id, r.score) for r in results]  # chunk_id, cosine_score
```

This **replaces `get_all_chunks()` entirely** for the retrieval path. Qdrant payload deliberately stores *only filter fields* (doc_id, language, page, is_bridge) — never full text, since Mongo remains the text source of truth and you don't want two places to keep in sync for content.

`.env` additions (yours, kept as-is, just confirming the connection point):

```bash
VECTOR_DB=qdrant
QDRANT_ENABLED=true
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=change-this-to-a-long-random-secret
QDRANT_COLLECTION=documents
QDRANT_TIMEOUT=30
QDRANT_PREFER_GRPC=false
QDRANT_VERIFY_SSL=true
QDRANT_RECREATE_COLLECTION=false
# QDRANT_VECTOR_SIZE is now DERIVED from USE_GPU, not set manually — see qdrant_client.py
QDRANT_DISTANCE=Cosine
```

I changed `QDRANT_VECTOR_SIZE` from a manually-set value to **derived from `USE_GPU`** — manually setting it risks drift if someone flips `USE_GPU` without remembering to update the dimension, which would break collection creation outright.

---

### 3.4 `core/retrieval_pipeline.py` — the shared pipeline (fixes Issue 5)

```python
def retrieve(query: str, doc_id_filter: str = None, top_k_final: int = 5) -> dict:
    # 1. Query preprocessing + language detection
    processed = QueryProcessor.process(query)
    query_language = detect_query_language(query)  # ta / en / mixed — see 3.5
    variants = processed.variants  # filler-removed + paraphrased variants

    # 2. Multi-query embedding
    query_vectors = embed_batch([query] + variants, is_query=True)

    # 3. Qdrant search — replaces "load all chunks"
    candidate_ids = set()
    vector_scores = {}
    for qv in query_vectors:
        for chunk_id, score in qdrant_search(qv, top_k=200, doc_id_filter=doc_id_filter):
            candidate_ids.add(chunk_id)
            vector_scores[chunk_id] = max(vector_scores.get(chunk_id, 0), score)

    # 4. Hydrate text + metadata from Mongo — ONLY for candidates, not all chunks
    candidates = mongo_get_chunks_by_ids(list(candidate_ids))  # ~200, not 5000+

    # 5. Stage 1: Hybrid scoring (vector + BM25 + section boost) — existing logic, reused
    hybrid_scored = hybrid_score(candidates, query, vector_scores)

    # 6. Stage 2: RRF re-rank — existing logic, reused
    reranked = rrf_rerank(hybrid_scored, query)

    # 7. Stage 3 [NEW]: MMR dedup — fixes Issue 7
    diversified = mmr_dedup(reranked, lambda_param=0.7, top_k=top_k_final)

    # 8. Relevance gating — existing logic, unchanged
    gated = apply_relevance_gate(diversified, min_score=0.35)

    return {
        "chunks": gated,
        "query_language": query_language,
    }
```

**MMR dedup** (Issue 7):

```python
import numpy as np

def mmr_dedup(scored_chunks, lambda_param=0.7, top_k=5):
    """Maximal Marginal Relevance — penalise chunks too similar to already-picked ones."""
    if len(scored_chunks) <= top_k:
        return scored_chunks

    selected = [scored_chunks[0]]
    remaining = scored_chunks[1:]

    while len(selected) < top_k and remaining:
        best_idx, best_mmr = 0, -float("inf")
        for i, candidate in enumerate(remaining):
            relevance = candidate["score"]
            max_sim = max(
                cosine_similarity(candidate["embedding"], s["embedding"])
                for s in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr, best_idx = mmr_score, i
        selected.append(remaining.pop(best_idx))

    return selected

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

`lambda_param=0.7` weights relevance over diversity 70/30 — tune downward (e.g. 0.5) if you still see redundant chunks in practice.

Both `api/semantic_search.py` and `voice/conversation_engine.py` now call `retrieve()` — neither implements its own retrieval logic.

---

### 3.5 Language-matched answer generation

This is the part that directly answers your "answer in the language I asked" requirement.

```python
def detect_query_language(query: str) -> str:
    """More nuanced than chunk-level detection — also catches Tanglish patterns."""
    tamil_ratio = len(TAMIL_RANGE.findall(query)) / max(len(query), 1)
    if tamil_ratio > 0.3:
        return "tamil_script"
    
    # Tanglish heuristic: Latin script but common Tamil function words transliterated
    tanglish_markers = ["eppadi", "enna", "epdi", "pannanum", "irukku", "vendam", "venum", "illa"]
    lowered = query.lower()
    if any(marker in lowered for marker in tanglish_markers):
        return "tanglish"
    
    return "english"


def build_answer_prompt(query: str, chunks: list, query_language: str) -> str:
    language_instruction = {
        "tamil_script": "பயனர் தமிழில் கேட்டுள்ளார். தமிழில் மட்டும் பதிலளிக்கவும்.",
        "tanglish": "The user asked in Tanglish (Tamil words in English script). Respond in the same Tanglish style — Tamil sentence structure/words in Latin script, matching their tone. Do not switch to pure Tamil script or formal English.",
        "english": "Respond in English.",
    }[query_language]

    context = "\n\n".join(
        f"[Source: {c['doc_title']}, Section: {c.get('parent_heading', 'N/A')}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    return f"""{language_instruction}

IMPORTANT: Answer using ONLY the information in the sources below. Cite the section and page for every claim. If the sources don't contain the answer, say so in the same language as the question — do not guess.

Sources:
{context}

Question: {query}

Answer:"""
```

The Tanglish branch is the trickiest one in practice — Mistral-7B will sometimes "correct" Tanglish into formal Tamil or English anyway. **Set expectations**: this prompt gets you most of the way, but a generic LLM faithfully producing natural Tanglish output is inherently inconsistent across models. If output quality here matters a lot, this is the strongest argument for moving to a Tamil-region model (Sarvam, Airavata) sooner rather than later — they're trained on more code-mixed Indian-language data and handle this far more naturally.

---

## 4. Issue → Fix cross-reference

| Issue | Severity | Fixed by |
|---|---|---|
| 1 — Chunks lose heading context | 🔴 Critical | §3.1 heading-aware chunking |
| 2 — All chunks loaded per query | 🔴 Critical | §3.3 Qdrant + Mongo hydrate-by-id |
| 3 — Embedding model can't handle Tamil | 🟡 Medium | §3.2 multilingual-e5-base / bge-m3 |
| 4 — No cross-page overlap | 🟡 Medium | §3.1 bridge chunks |
| 5 — Voice RAG duplicates/skips logic | 🟡 Medium | §3.4 shared `retrieval_pipeline.py` |
| 6 — Thin chunk metadata | 🟡 Medium | §3.1 `make_chunk()` enrichment |
| 7 — No dedup of near-identical chunks | 🟢 Low | §3.4 MMR dedup |
| *(new)* Answers don't match query language | — | §3.5 language detection + prompt |

---

## 5. Kubernetes / ops notes

- **Qdrant deployment**: standard StatefulSet + PVC for the collection data; expose only as a ClusterIP service (`qdrant:6333`) — no need for Ingress unless you want external debugging access. Keep `QDRANT_API_KEY` in a k8s Secret, not the ConfigMap.
- **Resource sizing**: at 100k chunks × 768-dim float32 vectors ≈ 300MB raw vector data — trivially small for Qdrant; default resource requests (512Mi–1Gi RAM, 0.5 CPU) are more than enough at this scale.
- **Backfill job**: re-embedding 10k–100k chunks on CPU with `multilingual-e5-base` — budget roughly 1.5–4 hours depending on actual chunk count and pod CPU; run as a k8s `Job`, not inline in a request handler.
- **`QDRANT_RECREATE_COLLECTION`**: keep `false` in normal operation. Only flip to `true` intentionally during a planned re-index (model swap, dimension change) — see migration runbook below.

---

## 6. Migration runbook (run once, in order)

1. Deploy Qdrant to k8s, confirm `http://qdrant:6333/healthz` responds inside the cluster.
2. Deploy new code with `core/chunking.py`, `core/embeddings.py`, `core/retrieval_pipeline.py`, Qdrant client — but **do not switch reads over yet**.
3. Run backfill job: re-process all existing documents through the new chunker → write to Mongo (new chunks collection or new fields) → embed with `multilingual-e5-base` → upsert to Qdrant.
4. Spot-check: pull 10–15 known queries (mix of Tamil, English, Tanglish), compare old vs new retrieval results manually.
5. Switch `api/semantic_search.py` and `voice/conversation_engine.py` to call `retrieval_pipeline.retrieve()`.
6. Monitor `MIN_CONTEXT_SCORE` gate rejection rate for a few days — if it spikes, your score distributions shifted with the new model and the threshold may need retuning (it was calibrated for the old model+scoring).
7. Decommission old `get_all_chunks()` path once confidence is high.

**When you eventually flip `USE_GPU=true`:** this is a second, separate re-index (new dimension, new model) — repeat steps 3–4 against a fresh Qdrant collection at 1024-dim, then cut over. Don't attempt to mix 768-dim and 1024-dim vectors in one collection.

---

## 7. Open items for you to confirm before implementation starts

1. **MIN_CONTEXT_SCORE recalibration** — the threshold of 0.35 was tuned against the old model/scoring combo. Expect to re-tune empirically after switching embedding models.
2. **Tanglish quality bar** — confirm whether Mistral-7B's Tanglish output is "good enough" for now, or whether this should accelerate the LLM swap to a Tamil-region model.
3. **Bridge chunk weighting** — decide if cross-page bridge chunks should be slightly down-weighted in scoring (they're somewhat redundant with neighbors by design).