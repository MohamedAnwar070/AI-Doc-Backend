import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SECTION_RE = re.compile(
    r"\b(?:section|sec\.?|s\.?)[\s:-]*([0-9]+[a-zA-Z-]*)\b", re.IGNORECASE
)
NUMBERED_HEADING_RE = re.compile(r"^\s*([0-9]+[a-zA-Z-]*)\.\s+(.+)$")

#: Default separator for boolean-AND queries (e.g. ``market + nmv``).
AND_SEPARATOR = "+"
#: Strong score boost applied to chunks that contain *every* term in an
#: AND query. The value is intentionally large (relative to the
#: per-component score weights in :func:`hybrid_score`) so that a chunk
#: with all ``+``-separated terms always sorts above a chunk with only
#: one of them, regardless of how high the lexical / vector scores are.
AND_FULL_HIT_BOOST: float = 1.5
#: Smaller boost when a chunk contains *some* (but not all) AND terms.
AND_PARTIAL_HIT_BOOST: float = 0.05

#: Minimum relevance score for a chunk to be fed to the LLM.
#: Chunks below this threshold are returned in matches but excluded
#: from the LLM context to avoid hallucinated answers.
MIN_CONTEXT_SCORE: float = 0.35

SUMMARY_TERMS = frozenset({
    "summary", "summarize", "summarise", "explain", "overview", "brief",
    "gist", "document", "file", "இந்த", "ஆவணம்",
})
#: Identity / personal-name questions (resumes, applications, title pages).
IDENTITY_TERMS = frozenset({
    "name", "named", "names", "fullname", "who", "candidate", "applicant",
    "employee", "person", "resume", "cv", "பெயர்", "யார", "யார்",
})
IDENTITY_QUERY_RE = re.compile(
    r"(?ix)\b("
    r"what(?:'s|\s+is)?\s+(?:her|his|their|the|this|that)?\s*name|"
    r"who\s+is\s+(?:this|she|he|the|that)|"
    r"(?:her|his|their|candidate|applicant|person(?:'s)?|full)\s+name|"
    r"name\s+of\s+(?:the\s+)?(?:person|candidate|applicant|employee)|"
    r"பெயர்\s*என்ன|யாரு|யார்\s*இது"
    r")\b"
)
#: Section headers that must not be mistaken for personal names.
_RESUME_SECTION_BLOCKLIST = frozenset({
    "technical skills", "work experience", "education", "projects",
    "certifications", "areas of interest", "languages known", "contact",
    "internship", "objective", "summary", "skills", "experience",
    "profile", "declaration", "achievements", "hobbies",
})
_SKILL_TOKEN_BLOCKLIST = frozenset({
    "JAVA", "SCRIPT", "JAVASCRIPT", "PYTHON", "REACT", "HTML", "CSS",
    "WEB", "DEVELOPMENT", "SKILLS", "PROJECTS", "TECHNICAL", "ENGINEER",
    "ENGINEERING", "MANAGEMENT", "SYSTEM", "SYSTEMS", "DATABASE", "SQL",
    "CLOUD", "AWS", "NODE", "ANGULAR", "SPRING", "BOOT", "MACHINE",
    "LEARNING", "DATA", "SCIENCE", "UIUX", "INTERNSHIP",
})
COMPARE_TERMS = frozenset({"compare", "difference", "different", "versus", "vs"})
BOUNDARY_TERMS = frozenset({
    "north", "south", "east", "west", "northern", "southern", "eastern",
    "western", "boundary", "boundaries", "bounded", "எல்லை", "வடக்கு",
    "தெற்கு", "கிழக்கு", "மேற்கு",
})
SURVEY_TERMS = frozenset({
    "survey", "surveys", "sy", "sno", "sno.", "s.no", "s.no.", "sf",
    "sfno", "surveyno", "patta", "subdivision", "sub", "division",
    "புல", "சர்வே",
})
LEGAL_FIELD_TERMS = frozenset({
    "schedule", "property", "extent", "door", "old", "new", "patta",
    "ward", "block", "village", "taluk", "district", "plot", "flat",
    "கதவு", "பட்டா", "சொத்து",
}) | BOUNDARY_TERMS | SURVEY_TERMS

LEGAL_FIELD_RE = re.compile(
    r"(?ix)\b("
    r"survey\s*(?:no|number)?|s\s*\.?\s*no\.?|sy\s*\.?\s*no\.?|sf\s*\.?\s*no\.?|"
    r"north|south|east|west|boundary|boundaries|bounded|schedule|property|"
    r"extent|door\s*no|patta|sub[-\s]*division|ward|block|village|taluk|district|"
    r"வடக்கு|தெற்கு|கிழக்கு|மேற்கு|எல்லை|சர்வே|பட்டா|சொத்து"
    r")\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# Query Processor — normalisation, key-term extraction, variant generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedQuery:
    """Result of query preprocessing."""
    original: str
    cleaned: str              # stripped of filler words
    key_terms: List[str]      # meaningful nouns/entities
    variants: List[str]       # expanded embedding variants
    word_count: int           # number of meaningful words
    intent: str               # summary | field_lookup | boundary_lookup | survey_lookup | section | factual | compare | keyword


class QueryProcessor:
    """Normalise, expand, and classify user queries for better retrieval.

    Production retrieval systems never embed the raw user query as-is.
    Short or vague queries (e.g. "state mentioned?") produce noisy
    embeddings because filler words dominate the signal.  This processor:

    1. **Strips filler words** that add no retrieval value.
    2. **Extracts key terms** — the meaningful nouns/entities.
    3. **Generates expanded variants** for multi-query embedding.
    4. **Classifies intent** (section lookup vs factual vs keyword).
    """

    #: Words that are common in conversational queries but add noise
    #: to embedding / lexical retrieval. Pronouns + ASR disfluencies included
    #: so voice queries do not rank ``he``/``we`` equal to content terms.
    FILLER_WORDS = frozenset({
        # English question / conversational fillers
        "mentioned", "about", "regarding", "related", "tell", "me",
        "please", "what", "is", "the", "any", "at", "are", "was",
        "were", "be", "been", "have", "has", "had", "do", "does",
        "did", "can", "could", "would", "should", "will", "may",
        "a", "an", "of", "in", "to", "for", "with", "on", "by",
        "from", "and", "or", "but", "not", "this", "that", "it",
        "its", "how", "who", "when", "where", "which", "there",
        "here", "also", "so", "then", "than", "very", "just",
        "only", "some", "no", "yes", "all", "each", "every",
        # Pronouns (critical for voice — otherwise BM25 weights them like content)
        "i", "you", "he", "she", "we", "they", "him", "her", "them",
        "his", "hers", "our", "ours", "your", "yours", "their", "theirs",
        "my", "mine", "myself", "yourself", "himself", "herself",
        "itself", "ourselves", "themselves",
        # Discourse / ASR disfluencies
        "okay", "ok", "well", "right", "um", "uh", "erm", "like",
        "exactly", "actually", "basically", "literally", "really",
        "something", "things", "stuff", "kinda", "sorta", "yeah", "yep", "nah",
        "alright", "gonna", "wanna", "gotta",
        # Tamil / Tanglish fillers
        "enna", "ennanu", "sollu", "solunga", "pathu", "pathi",
        "pathina", "iruku", "kidaikuma", "seri", "appo", "ippo",
        "romba", "konjam", "illa", "illai", "aama", "sari",
    })

    #: Domain tokens that must never be stripped as fillers.
    _PROTECTED_TERMS = LEGAL_FIELD_TERMS | SURVEY_TERMS | BOUNDARY_TERMS | IDENTITY_TERMS | frozenset({
        "number", "nos",
    })

    def _is_filler(self, token: str, tokens: Sequence[str], index: int) -> bool:
        """Return True if token should be dropped (respecting domain guards)."""
        if len(token) < 2:
            return True
        if token in self._PROTECTED_TERMS:
            return False
        # Keep "no" when adjacent to survey vocabulary (survey no / s no / sy no)
        if token == "no":
            prev_t = tokens[index - 1] if index > 0 else ""
            next_t = tokens[index + 1] if index + 1 < len(tokens) else ""
            if prev_t in SURVEY_TERMS or next_t.isdigit() or prev_t in {
                "s", "sy", "sf", "sno", "survey"
            }:
                return False
        return token in self.FILLER_WORDS

    def process(self, raw_query: str) -> ProcessedQuery:
        """Run the full preprocessing pipeline on a raw user query."""
        original = raw_query.strip()
        # Strip trailing punctuation for cleaner matching
        base = re.sub(r"[?!.]+$", "", original).strip()

        # Detect intent first (before stripping words)
        intent = self._classify_intent(base)

        # Tokenise and strip fillers (domain-protected)
        tokens = re.findall(r"\w+", base.lower())
        key_terms = [
            t for i, t in enumerate(tokens)
            if not self._is_filler(t, tokens, i)
        ]

        # If aggressive stripping removed everything, fall back to all tokens
        if not key_terms:
            key_terms = [t for t in tokens if len(t) >= 2]

        # Identity questions often clean to bare "name" which never appears in
        # the document — expand so lexical/vector paths can still hit headers.
        if intent == "identity_lookup":
            for extra in ("name", "full", "candidate", "person"):
                if extra not in key_terms:
                    key_terms.append(extra)

        cleaned = " ".join(key_terms) if key_terms else base
        variants = self._build_variants(key_terms, base, intent)

        return ProcessedQuery(
            original=original,
            cleaned=cleaned,
            key_terms=key_terms,
            variants=variants,
            word_count=len(key_terms),
            intent=intent,
        )

    def _classify_intent(self, query: str) -> str:
        """Classify the query intent for scoring strategy."""
        if SECTION_RE.search(query):
            return "section_lookup"
        if IDENTITY_QUERY_RE.search(query):
            return "identity_lookup"
        tokens = re.findall(r"\w+", query.lower())
        meaningful = [
            t for i, t in enumerate(tokens)
            if not self._is_filler(t, tokens, i)
        ]
        terms = set(meaningful)
        if terms & COMPARE_TERMS:
            return "compare"
        # Summary before field_lookup so "summarize this property document"
        # keeps the page-chunk summary boost (F10).
        if terms and terms <= SUMMARY_TERMS:
            return "summary"
        if any(t in terms for t in ("summarize", "summarise", "summary", "overview", "brief", "gist", "explain")):
            return "summary"
        # "name" / "who" alone (after filler strip of her/his) → identity
        if terms & {"name", "named", "names", "who", "பெயர்"}:
            return "identity_lookup"
        if (terms & BOUNDARY_TERMS) and (terms & SURVEY_TERMS):
            return "boundary_lookup"
        if terms & BOUNDARY_TERMS:
            return "boundary_lookup"
        if terms & SURVEY_TERMS:
            return "survey_lookup"
        if terms & LEGAL_FIELD_TERMS:
            return "field_lookup"
        if len(meaningful) <= 2:
            return "keyword"
        return "factual"

    def _build_variants(self, key_terms: List[str], original: str, intent: str) -> List[str]:
        """Generate query variants for multi-query embedding.

        Each variant captures a different "angle" of the user's intent.
        The retrieval layer embeds all variants and takes the max score
        per chunk — a lightweight version of HyDE without LLM calls.
        """
        variants: List[str] = []

        # Variant 1: The cleaned key-terms query
        cleaned = " ".join(key_terms)
        if cleaned:
            variants.append(cleaned)

        # Variant 2: Original query (in case filler words carry context)
        stripped_original = re.sub(r"[?!.]+$", "", original).strip()
        if stripped_original and stripped_original.lower() != cleaned.lower():
            variants.append(stripped_original)

        # Variant 3: For keyword intent, try each key term solo (only if few terms)
        if intent == "keyword" and 1 <= len(key_terms) <= 3:
            for term in key_terms:
                if term not in variants:
                    variants.append(term)

        # Variant 4: Bigram combinations for multi-term queries
        if len(key_terms) >= 2:
            for i in range(len(key_terms) - 1):
                bigram = f"{key_terms[i]} {key_terms[i + 1]}"
                if bigram not in variants:
                    variants.append(bigram)

        variants.extend(self._legal_expansions(key_terms, intent))
        variants.extend(self._identity_expansions(intent))

        # Cap at 5 to limit embedding API calls
        return list(dict.fromkeys(v for v in variants if v))[:6]

    def _identity_expansions(self, intent: str) -> List[str]:
        if intent != "identity_lookup":
            return []
        return [
            "full name",
            "candidate name",
            "person name",
            "applicant name",
        ]

    def _legal_expansions(self, key_terms: List[str], intent: str) -> List[str]:
        terms = set(key_terms)
        variants: List[str] = []
        if intent in {"boundary_lookup", "survey_lookup", "field_lookup"}:
            if terms & BOUNDARY_TERMS:
                variants.append("property boundary schedule")
            if terms & SURVEY_TERMS:
                variants.append("survey number property schedule")
            if {"north", "survey"} <= terms:
                variants.append("north boundary survey number")
            if "schedule" in terms or "property" in terms:
                variants.append("schedule of property boundaries survey")
        return variants


def _looks_like_person_name_line(line: str) -> bool:
    """True for resume name lines like ``KALAI PRIYA R`` / ``Kalai Priya R``."""
    s = (line or "").strip().rstrip(":")
    if not s or len(s) < 3 or len(s) > 48:
        return False
    if s.lower() in _RESUME_SECTION_BLOCKLIST:
        return False
    # Title Case or ALL CAPS, 2–4 tokens, no digits
    if re.search(r"\d", s):
        return False
    if not re.match(r"^[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){1,3}$", s):
        return False
    words = s.split()
    if any(w.upper() in _SKILL_TOKEN_BLOCKLIST for w in words):
        return False
    # Prefer lines that are mostly letters (names), not long prose
    if sum(len(w) for w in words) > 36:
        return False
    # At least one word length >= 3 (avoid "AI R")
    if not any(len(w) >= 3 for w in words):
        return False
    return True


def identity_chunk_boost(text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
    """Boost chunks that look like they contain a personal name header.

    Resumes often print the candidate name in ALL CAPS without the word
    ``name``, so pure keyword search for "name" fails. Prefer page-1 /
    contact-style chunks with name-shaped lines.
    """
    if not text:
        return 0.0
    metadata = metadata or {}
    boost = 0.0
    page = metadata.get("page")
    try:
        page_i = int(page) if page is not None else 1
    except (TypeError, ValueError):
        page_i = 1
    if page_i <= 1:
        boost += 0.12

    head = text[:800]
    if any(_looks_like_person_name_line(line) for line in head.splitlines()[:20]):
        boost += 0.30
    lower = head.lower()
    if any(k in lower for k in ("contact", "email", "linkedin", "resume", "cv", "objective")):
        boost += 0.08
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", head):
        boost += 0.06
    return min(boost, 0.45)


@dataclass(frozen=True)
class SearchChunk:
    doc_id: str
    file_path: str
    chunk_id: int
    text: str
    embed_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchMatch:
    score: float
    chunk: SearchChunk


def normalize_ocr_text(text: str, preserve_tabs: bool = False) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\xa0", " ")
    if preserve_tabs:
        text = re.sub(r" +", " ", text)
    else:
        text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?i)\bsectio[nm]\b", "section", text)
    text = re.sub(r"(?i)\bsecti[o0]n\b", "section", text)
    text = re.sub(r"(?i)\bs\s*\.\s*([0-9])", r"section \1", text)
    text = re.sub(r"(?i)\bsy\s*\.?\s*no\.?\b", "Survey No", text)
    text = re.sub(r"(?i)\bs\s*\.?\s*no\.?\b", "Survey No", text)
    text = re.sub(r"(?i)\bsurvey\s*(?:number|no\.?)\b", "Survey No", text)
    text = re.sub(r"(?i)\bsub\s*[-.]?\s*division\b", "sub-division", text)
    return text.strip()


def extract_section_label(text: str) -> Optional[str]:
    sample_lines = [line.strip() for line in text.splitlines() if line.strip()][:8]

    for line in sample_lines:
        section_match = SECTION_RE.search(line)
        if section_match:
            return f"Section {section_match.group(1)}"

        numbered_match = NUMBERED_HEADING_RE.match(line)
        if numbered_match:
            return f"Section {numbered_match.group(1)}"

    return None


def extract_query_section_refs(query: str) -> List[str]:
    refs = [m.group(1).lower() for m in SECTION_RE.finditer(query)]
    if refs:
        return list(dict.fromkeys(refs))

    numeric_refs = re.findall(r"\b([0-9]+[a-zA-Z-]*)\b", query)
    if any(word.lower() in query.lower() for word in ("section", "sec", "s.")):
        return list(dict.fromkeys(ref.lower() for ref in numeric_refs))

    return []


def split_flat_text_pages(text: str, preserve_tabs: bool = False) -> List[Tuple[Optional[int], str]]:
    normalized = normalize_ocr_text(text, preserve_tabs=preserve_tabs)
    if not normalized:
        return []

    page_matches = list(
        re.finditer(r"^--- Page\s+(\d+)\s+---$", normalized, re.MULTILINE)
    )
    if not page_matches:
        return [(None, normalized)]

    pages: List[Tuple[Optional[int], str]] = []
    for index, match in enumerate(page_matches):
        page_num = int(match.group(1))
        start = match.end()
        end = (
            page_matches[index + 1].start()
            if index + 1 < len(page_matches)
            else len(normalized)
        )
        page_text = normalized[start:end].strip()
        if page_text:
            pages.append((page_num, page_text))
    return pages


def split_into_blocks(text: str, preserve_tabs: bool = False) -> List[str]:
    normalized = normalize_ocr_text(text, preserve_tabs=preserve_tabs)
    if not normalized:
        return []

    blocks = [
        block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()
    ]
    if blocks:
        return blocks

    return [line.strip() for line in normalized.splitlines() if line.strip()]


#: Abbreviations that should NOT trigger sentence splits (e.g. "Dr.", "No.", "Rs.").
_ABBREV_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|No|Nos|Rs|Sy|St|Govt|Dept|Ref|Vol|Ch|Sec|Art|Ltd|Inc|Corp|approx|viz|i\.e|e\.g)\.",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at natural boundaries.

    Handles:
    - Standard terminators: ``. ? !``
    - Tamil/Devanagari terminators: ``। ॥``
    - Abbreviation safety (Dr., No., Rs., Sy., etc.)
    - Decimal number safety (1.5, Rs.10,000)
    - Newlines as natural sentence boundaries
    """
    if not text or not text.strip():
        return []

    _PLACEHOLDER = "\x00ABBRDOT\x00"
    _NUM_PLACEHOLDER = "\x00NUMDOT\x00"

    # Step 1: Protect abbreviation dots using re.sub (safe, single-pass)
    protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _PLACEHOLDER, text)

    # Step 2: Protect decimal number dots (e.g. 1.5, 10.25)
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_NUM_PLACEHOLDER}\2", protected)

    # Step 3: Split at sentence terminators followed by whitespace, or at newlines
    raw_sentences = re.split(
        r"(?<=[.!?।॥])\s+|(?<=\n)",
        protected,
    )

    # Step 4: Restore protected dots and collect non-empty sentences
    sentences: List[str] = []
    for s in raw_sentences:
        restored = s.replace(_PLACEHOLDER, ".").replace(_NUM_PLACEHOLDER, ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Algorithm:
    1. Split into paragraphs (``\\n\\n``) then into sentences.
    2. Greedily merge sentences until ``chunk_size`` is reached.
    3. Overlap uses the last N whole sentences that fit within ``overlap`` chars.
    4. A single sentence longer than ``chunk_size`` is kept whole (never split mid-sentence).

    The function signature is backwards-compatible with the original paragraph-only
    splitter — all existing callers work without modification.
    """
    blocks = split_into_blocks(text)
    if not blocks:
        return []

    # Flatten all paragraphs into a list of (sentence_text, paragraph_break_before)
    all_sentences: List[str] = []
    for i, block in enumerate(blocks):
        sents = _split_sentences(block)
        if not sents:
            continue
        if all_sentences and sents:
            # Insert a paragraph-break marker between blocks
            all_sentences.append("\n")
        all_sentences.extend(sents)

    if not all_sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in all_sentences:
        is_para_break = sentence == "\n"
        if is_para_break:
            # Just mark a paragraph transition without adding text length
            if current_sentences:
                current_sentences.append(sentence)
            continue

        added_len = len(sentence) + (2 if current_sentences else 0)  # " " or "\n\n" joiner

        if current_len + added_len > chunk_size and current_sentences:
            # Emit the current chunk
            chunk_text_str = _join_sentences(current_sentences)
            chunks.append(chunk_text_str)

            # Compute sentence-boundary overlap: take trailing sentences that fit within `overlap`
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if s == "\n":
                    continue
                if overlap_len + len(s) + 2 > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 2

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s in current_sentences if s != "\n") + max(0, (len(current_sentences) - 1) * 2)

        current_sentences.append(sentence)
        current_len += added_len

    # Emit the final chunk
    if current_sentences:
        chunk_text_str = _join_sentences(current_sentences)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)

    return chunks


def _join_sentences(sentences: List[str]) -> str:
    """Join sentence fragments, respecting paragraph break markers."""
    parts: List[str] = []
    for s in sentences:
        if s == "\n":
            # Mark a paragraph boundary
            if parts and not parts[-1].endswith("\n\n"):
                parts.append("\n\n")
        else:
            if parts and parts[-1] != "\n\n" and not parts[-1].endswith("\n\n"):
                parts.append(" ")
            parts.append(s)
    return "".join(parts).strip()


def _make_chunk(
    *,
    doc_id: str,
    file_path: str,
    chunk_id: int,
    text: str,
    filename: Optional[str],
    base_metadata: Optional[Dict[str, Any]] = None,
    embed_prefix: Optional[str] = None,
) -> SearchChunk:
    metadata = dict(base_metadata or {})
    if filename:
        metadata.setdefault("filename", filename)
    metadata.setdefault("section_label", extract_section_label(text))
    metadata.setdefault("chunk_type", "semantic")
    metadata.setdefault("field_hits", extract_legal_field_hits(text))
    from core.doc_metadata import detect_document_language, extract_title
    metadata.setdefault("language", detect_document_language(text))
    if "title" not in metadata and filename:
        metadata["title"] = extract_title(text, filename)
    clean_prefix = normalize_ocr_text(embed_prefix or "")
    embed_text = f"{clean_prefix}\n\n{text}".strip() if clean_prefix else text
    return SearchChunk(
        doc_id=doc_id,
        file_path=file_path,
        chunk_id=chunk_id,
        text=text,
        embed_text=embed_text,
        metadata=metadata,
    )



#: Regex for recognizing numbered or bulleted list item starts in layout and text chunks.
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:"
    r"[-*•▪→►✓\u2022\u25E6]|"                  # Bullet symbols
    r"(?:\d{1,3}|[A-Za-z]|[ivxlcdm]{1,4})[.)\]]" # Numbered/lettered item (e.g. 1., a), ii.)
    r")\s+[^\s]",
    re.IGNORECASE,
)


def _looks_like_heading(element_type: str, text: str) -> bool:
    """Best-effort heading detection for layout chunks.

    OCR/layout engines do not always agree on element type names. This
    helper intentionally stays conservative: headings become context for
    the following chunk, but are still emitted as their own chunk when
    they contain enough useful text.
    """
    kind = (element_type or "").lower()
    if kind in {"heading", "title", "section_header", "header"}:
        return True

    sample = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not sample or len(sample) > 140:
        return False
    if SECTION_RE.search(sample):
        return True
    if NUMBERED_HEADING_RE.match(sample):
        if len(sample.split()) <= 5 or sample.upper() == sample:
            return True
    alpha = [ch for ch in sample if ch.isalpha()]
    return bool(alpha) and len(sample.split()) <= 12 and sample.upper() == sample


def _looks_like_list_item(element_type: str, text: str) -> bool:
    """Determine if a text segment represents a structured list item or bullet point."""
    kind = (element_type or "").lower()
    if kind in {"list", "list_item", "bullet", "item", "enumeration"}:
        return True
    if not text or len(text.strip()) > 1000:  # Massive paragraphs aren't single list markers
        return False
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return bool(_LIST_ITEM_RE.match(first_line))


class HierarchyTracker:
    """Multi-level structural hierarchy tracker for RAG document indexing.

    Maintains a hierarchical breadcrumb trail (Major Title / Section -> Subsection -> Clause)
    as documents are iterated across structured layout elements or text pages, resolving
    semantic fragmentation across deeply nested document sections.
    """
    def __init__(self) -> None:
        self.l1: Optional[str] = None
        self.l2: Optional[str] = None
        self.l3: Optional[str] = None

    def update(self, heading_text: str) -> int:
        """Register a newly detected heading and update structural tier."""
        clean = " ".join(heading_text.split()).strip()[:200]
        upper = clean.upper()
        # Major Level 1 Sections: All-caps titles, Schedules, Chapters, Parts
        if (clean == upper and len(clean) <= 70 and not any(c.isdigit() for c in clean)) or any(clean.lower().startswith(p) for p in ["schedule", "part ", "chapter ", "section ", "deed of"]):
            self.l1 = clean
            self.l2 = None
            self.l3 = None
            return 1
        # Level 2 Subsections: Numbered headers (e.g. 1.1, Clause 1, Item 1)
        elif any(clean.lower().startswith(p) for p in ["clause", "item ", "article "]) or re.match(r"^\d{1,2}\.\s+", clean):
            self.l2 = clean
            self.l3 = None
            return 2
        else:
            if not self.l1:
                self.l1 = clean
                return 1
            elif not self.l2:
                self.l2 = clean
                return 2
            else:
                self.l3 = clean
                return 3

    def get_parent_heading(self) -> Optional[str]:
        """Return immediate parent heading (for backwards compatibility)."""
        return self.l3 or self.l2 or self.l1

    def get_hierarchy_path(self) -> Optional[str]:
        """Return formatted full breadcrumb trail (e.g., 'SCHEDULE A > Item No. 1')."""
        parts = [h for h in (self.l1, self.l2, self.l3) if h]
        return " > ".join(parts) if parts else None


def _page_heading_context(
    filename: Optional[str],
    page_num: Optional[int],
    heading: Optional[str],
    hierarchy_path: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if filename:
        parts.append(f"Document: {filename}")
    if page_num:
        parts.append(f"Page: {page_num}")
    if hierarchy_path and hierarchy_path != heading:
        parts.append(f"Hierarchy: {hierarchy_path}")
    if heading:
        parts.append(f"Heading: {heading}")
    return "\n".join(parts)


def extract_legal_field_hits(text: str) -> List[str]:
    """Return normalised legal/property field terms present in text."""
    hits: List[str] = []
    for match in LEGAL_FIELD_RE.finditer(text or ""):
        hit = normalize_ocr_text(match.group(0)).lower().replace(" ", "_")
        hits.append(hit)
    return list(dict.fromkeys(hits))


def _chunk_type_for_text(text: str, default: str = "semantic", element_type: str = "paragraph") -> str:
    lower = (text or "").lower()
    hits = set(extract_legal_field_hits(lower))
    if "schedule" in lower and ("property" in lower or hits & {"survey_no", "north", "south", "east", "west"}):
        return "schedule"
    if hits:
        return "fact_window"
    if _looks_like_list_item(element_type, text):
        return "list_item"
    return default


def _fact_windows(page_text: str, radius: int = 2) -> List[str]:
    lines = [line.strip() for line in normalize_ocr_text(page_text).splitlines() if line.strip()]
    windows: List[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if not LEGAL_FIELD_RE.search(line):
            continue
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        window = "\n".join(lines[start:end]).strip()
        key = re.sub(r"\s+", " ", window.lower())
        if len(window) >= 20 and key not in seen:
            seen.add(key)
            windows.append(window)
    return windows


def _append_page_support_chunks(
    *,
    chunks: List[SearchChunk],
    chunk_id: int,
    doc_id: str,
    filename: Optional[str],
    page_num: Optional[int],
    page_text: str,
    source: str,
    file_path: str,
    heading: Optional[str] = None,
    hierarchy_path: Optional[str] = None,
) -> int:
    page_text = normalize_ocr_text(page_text)
    if len(page_text) <= 20:
        return chunk_id

    for window in _fact_windows(page_text):
        chunk_type = _chunk_type_for_text(window, default="fact_window")
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                file_path=file_path,
                chunk_id=chunk_id,
                text=window,
                filename=filename,
                base_metadata={
                    "page": page_num,
                    "source": source,
                    "parent_heading": heading,
                    "hierarchy_path": hierarchy_path,
                    "chunk_type": chunk_type,
                    "is_support_chunk": True,
                },
                embed_prefix=_page_heading_context(filename, page_num, heading, hierarchy_path),
            )
        )
        chunk_id += 1

    if heading and len(heading.strip()) >= 3:
        section_text = f"SECTION SUMMARY: {heading}\n\n{page_text[:1200]}".strip()
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                file_path=file_path,
                chunk_id=chunk_id,
                text=section_text,
                filename=filename,
                base_metadata={
                    "page": page_num,
                    "source": source,
                    "parent_heading": heading,
                    "hierarchy_path": hierarchy_path,
                    "chunk_type": "section",
                    "is_support_chunk": True,
                },
                embed_prefix=_page_heading_context(filename, page_num, heading, hierarchy_path),
            )
        )
        chunk_id += 1

    if len(page_text) <= 4000:
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                file_path=file_path,
                chunk_id=chunk_id,
                text=page_text,
                filename=filename,
                base_metadata={
                    "page": page_num,
                    "source": source,
                    "parent_heading": heading,
                    "hierarchy_path": hierarchy_path,
                    "chunk_type": "page",
                    "is_support_chunk": True,
                },
                embed_prefix=_page_heading_context(filename, page_num, heading, hierarchy_path),
            )
        )
        chunk_id += 1

    return chunk_id


def _add_parent_child_chunks(
    *,
    text: str,
    doc_id: str,
    file_path: str,
    filename: Optional[str],
    base_metadata: dict,
    embed_prefix: Optional[str],
    start_chunk_id: int,
    chunk_size: int = 800,
    overlap: int = 200,
    parent_chunk_size: int = 2400,
) -> Tuple[List[SearchChunk], int]:
    """Split text into parent (2400ch) and child (800ch) chunks with parent_chunk_id linking."""
    chunks: List[SearchChunk] = []
    chunk_id = start_chunk_id

    if len(text) > chunk_size * 2:
        parent_pieces = chunk_text(text, chunk_size=parent_chunk_size, overlap=400)
        for p_text in parent_pieces:
            p_chunk_id = chunk_id
            chunk_id += 1

            child_pieces = chunk_text(p_text, chunk_size=chunk_size, overlap=overlap)
            child_ids: List[int] = []

            for c_text in child_pieces:
                c_chunk_id = chunk_id
                chunk_id += 1
                child_ids.append(c_chunk_id)

                c_meta = {
                    **base_metadata,
                    "parent_chunk_id": p_chunk_id,
                    "parent_text": p_text,
                }
                chunks.append(
                    _make_chunk(
                        doc_id=doc_id,
                        file_path=file_path,
                        chunk_id=c_chunk_id,
                        text=c_text,
                        filename=filename,
                        base_metadata=c_meta,
                        embed_prefix=embed_prefix,
                    )
                )

            p_meta = {
                **base_metadata,
                "chunk_type": "parent",
                "child_chunk_ids": child_ids,
            }
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    file_path=file_path,
                    chunk_id=p_chunk_id,
                    text=p_text,
                    filename=filename,
                    base_metadata=p_meta,
                    embed_prefix=embed_prefix,
                )
            )
    else:
        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            c_meta = {
                **base_metadata,
                "parent_text": text if len(text) > len(piece) else None,
            }
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    file_path=file_path,
                    chunk_id=chunk_id,
                    text=piece,
                    filename=filename,
                    base_metadata=c_meta,
                    embed_prefix=embed_prefix,
                )
            )
            chunk_id += 1

    return chunks, chunk_id


def build_search_chunks(
    *,
    doc_id: str,
    filename: Optional[str],
    text: str,
    structured: Optional[dict] = None,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[SearchChunk]:
    chunks: List[SearchChunk] = []
    chunk_id = 0

    if structured and structured.get("pages"):
        tracker = HierarchyTracker()
        for page in structured.get("pages", []):
            page_num = page.get("page")
            page_parts: List[str] = []
            for element in page.get("elements", []):
                element_text = normalize_ocr_text(element.get("text", ""))
                if not element_text:
                    continue
                page_parts.append(element_text)

                element_type = element.get("type", "paragraph")
                is_heading = _looks_like_heading(element_type, element_text)
                if is_heading:
                    tracker.update(element_text)

                parent_heading = tracker.get_parent_heading() if not is_heading else None
                hierarchy_path = tracker.get_hierarchy_path() if not is_heading else None
                
                is_list = _looks_like_list_item(element_type, element_text)
                actual_element_type = "list_item" if is_list and not is_heading else element_type

                base_metadata = {
                    "page": page_num,
                    "element_type": actual_element_type,
                    "source": "structured",
                    "parent_heading": parent_heading,
                    "hierarchy_path": hierarchy_path,
                    "section": tracker.l1 if not is_heading else None,
                    "subsection": tracker.l2 if not is_heading else None,
                    "is_heading": is_heading,
                    "chunk_type": "table" if actual_element_type == "table" else _chunk_type_for_text(element_text, element_type=actual_element_type),
                }
                embed_prefix = _page_heading_context(
                    filename=filename,
                    page_num=page_num,
                    heading=parent_heading,
                    hierarchy_path=hierarchy_path,
                )

                if actual_element_type == "table":
                    chunks.append(
                        _make_chunk(
                            doc_id=doc_id,
                            file_path=f"storage://{doc_id}/result.json",
                            chunk_id=chunk_id,
                            text=element_text,
                            filename=filename,
                            base_metadata=base_metadata,
                            embed_prefix=embed_prefix,
                        )
                    )
                    chunk_id += 1
                    continue

                new_chunks, chunk_id = _add_parent_child_chunks(
                    text=element_text,
                    doc_id=doc_id,
                    file_path=f"storage://{doc_id}/result.json",
                    filename=filename,
                    base_metadata=base_metadata,
                    embed_prefix=embed_prefix,
                    start_chunk_id=chunk_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                chunks.extend(new_chunks)

            chunk_id = _append_page_support_chunks(
                chunks=chunks,
                chunk_id=chunk_id,
                doc_id=doc_id,
                filename=filename,
                page_num=page_num,
                page_text="\n".join(page_parts),
                source="structured",
                file_path=f"storage://{doc_id}/result.json",
                heading=tracker.get_parent_heading(),
                hierarchy_path=tracker.get_hierarchy_path(),
            )

        if chunks:
            return _finalize_chunks(chunks)

    tracker_flat = HierarchyTracker()
    for page_num, page_text in split_flat_text_pages(text):
        # Proactively check if the leading lines on flat text page represent a section heading
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if lines and _looks_like_heading("paragraph", lines[0]):
            tracker_flat.update(lines[0])

        new_chunks, chunk_id = _add_parent_child_chunks(
            text=page_text,
            doc_id=doc_id,
            file_path=f"storage://{doc_id}/result.txt",
            filename=filename,
            base_metadata={
                "page": page_num,
                "source": "flat",
                "parent_heading": tracker_flat.get_parent_heading(),
                "hierarchy_path": tracker_flat.get_hierarchy_path(),
                "section": tracker_flat.l1,
                "subsection": tracker_flat.l2,
                "chunk_type": _chunk_type_for_text(page_text),
            },
            embed_prefix=_page_heading_context(filename, page_num, tracker_flat.get_parent_heading(), tracker_flat.get_hierarchy_path()),
            start_chunk_id=chunk_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(new_chunks)

        chunk_id = _append_page_support_chunks(
            chunks=chunks,
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename,
            page_num=page_num,
            page_text=page_text,
            source="flat",
            file_path=f"storage://{doc_id}/result.txt",
            heading=tracker_flat.get_parent_heading(),
            hierarchy_path=tracker_flat.get_hierarchy_path(),
        )

    return _finalize_chunks(chunks)



def _deduplicate_chunks(chunks: List[SearchChunk]) -> List[SearchChunk]:
    """Remove exact and near-duplicate chunks within a single document.

    Uses normalised text hashing for exact matches, and Jaccard similarity
    on word sets for near-duplicates (threshold > 0.85). Keeps the first
    (and longest, in case of near-dupes) occurrence.
    """
    if len(chunks) <= 1:
        return chunks

    def _norm_key(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _word_set(text: str) -> set:
        return set(_norm_key(text).split())

    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    seen_exact: Dict[str, int] = {}  # norm_key -> index in deduped
    deduped: List[SearchChunk] = []
    deduped_word_sets: List[set] = []

    for chunk in chunks:
        key = _norm_key(chunk.text)
        is_support = chunk.metadata.get("is_support_chunk") or chunk.metadata.get("chunk_type") in {"section", "parent", "page"}
        if not key or is_support:
            deduped.append(chunk)
            deduped_word_sets.append(_word_set(chunk.text))
            continue

        # Exact duplicate check
        if key in seen_exact:
            continue

        # Near-duplicate check (only against recent chunks to keep O(n) manageable)
        ws = _word_set(chunk.text)
        is_near_dup = False
        lookback = min(len(deduped_word_sets), 20)  # Check last 20 chunks
        for prev_ws in deduped_word_sets[-lookback:]:
            if _jaccard(ws, prev_ws) > 0.85:
                is_near_dup = True
                break

        if is_near_dup:
            continue

        seen_exact[key] = len(deduped)
        deduped.append(chunk)
        deduped_word_sets.append(ws)

    return deduped


def _add_neighbour_links(chunks: List[SearchChunk]) -> None:
    """Set prev_chunk_id / next_chunk_id on each chunk's metadata for future context expansion."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["prev_chunk_id"] = chunks[i - 1].chunk_id if i > 0 else None
        chunk.metadata["next_chunk_id"] = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None


def _finalize_chunks(chunks: List[SearchChunk]) -> List[SearchChunk]:
    """Post-process generated chunks: deduplicate and add neighbour links."""
    deduped = _deduplicate_chunks(chunks)
    _add_neighbour_links(deduped)
    return deduped


def build_corpus(
    storage: Any,
    chunk_size: int = 800,
    overlap: int = 200,
    job_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[SearchChunk]:
    chunks: List[SearchChunk] = []
    job_ids = [job_id] if job_id else storage.list_jobs()

    for jid in job_ids:
        if doc_id and jid != doc_id:
            continue

        status = storage.load_status(jid) or {}
        filename = status.get("pdf_filename")
        structured = (
            storage.read_structured_result(jid)
            if hasattr(storage, "read_structured_result")
            else None
        )
        text = storage.read_result(jid)
        if not text and not structured:
            continue

        chunks.extend(
            build_search_chunks(
                doc_id=jid,
                filename=filename,
                text=text or "",
                structured=structured,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks


def parse_and_terms(query: str, separator: str = AND_SEPARATOR) -> List[str]:
    """Return the list of AND-separated query segments.

    A query of ``"market + nmv"`` returns ``["market", "nmv"]``. A query
    without the separator returns an empty list (the caller should fall
    back to the regular ranking path). Whitespace around the separator
    is ignored.
    """
    if not query or separator not in query:
        return []
    parts = [segment.strip() for segment in query.split(separator)]
    return [segment for segment in parts if segment]


def _and_term_hits(text: str, terms: Sequence[str]) -> int:
    """Count how many AND terms appear (case-insensitively) in ``text``."""
    if not terms or not text:
        return 0
    haystack = text.lower()
    return sum(1 for term in terms if term.lower() in haystack)


def _keyword_score(query: str, text: str) -> float:
    q = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
    if not q:
        return 0.0
    t = text.lower()
    hits = sum(1 for w in q if w in t)
    return hits / max(1, len(q))


def _build_df_map(texts: Sequence[str]) -> Tuple[Dict[str, int], int]:
    """Document frequency map over a candidate corpus (one doc = one text)."""
    df: Dict[str, int] = defaultdict(int)
    n = 0
    for text in texts:
        terms = set(re.findall(r"\w+", (text or "").lower()))
        if not terms:
            continue
        n += 1
        for t in terms:
            if len(t) >= 2:
                df[t] += 1
    return df, max(n, 1)


def _bm25_idf(term: str, df_map: Optional[Dict[str, int]], n_docs: int) -> float:
    """Standard BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))."""
    if not df_map or n_docs <= 0:
        return 1.0
    df = df_map.get(term, 0)
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _bm25_score(
    query: str,
    text: str,
    avg_doc_len: float = 300.0,
    k1: float = 1.5,
    b: float = 0.75,
    df_map: Optional[Dict[str, int]] = None,
    n_docs: int = 0,
) -> float:
    """Okapi BM25 scoring — TF saturation + length norm + IDF.

    Without IDF, common pronouns and content terms weigh the same (F7).
    Pass corpus ``df_map`` / ``n_docs`` when scoring a candidate set.
    """
    query_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
    if not query_terms:
        return 0.0

    doc_terms = re.findall(r"\w+", text.lower())
    doc_len = len(doc_terms)
    if doc_len == 0:
        return 0.0

    tf_map: Dict[str, int] = defaultdict(int)
    for t in doc_terms:
        tf_map[t] += 1

    score = 0.0
    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf = _bm25_idf(term, df_map, n_docs)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
        score += idf * (numerator / denominator)

    # Average over query terms so long queries are comparable
    return score / max(1, len(query_terms))


def _exact_phrase_score(query: str, text: str) -> float:
    compact_query = normalize_ocr_text(query).lower()
    compact_text = normalize_ocr_text(text).lower()
    if not compact_query or not compact_text:
        return 0.0
    if compact_query in compact_text:
        return 1.0
    return 0.0


def _section_boost(query: str, text: str, metadata: Dict[str, Any]) -> float:
    refs = extract_query_section_refs(query)
    if not refs:
        return 0.0

    section_label = (metadata.get("section_label") or "").lower()
    text_lower = text.lower()
    score = 0.0

    for ref in refs:
        if ref in section_label:
            score = max(score, 1.0)
        elif re.search(rf"\bsection\s*{re.escape(ref)}\b", text_lower):
            score = max(score, 0.9)
        elif re.search(rf"^\s*{re.escape(ref)}\.\s", text_lower, re.MULTILINE):
            score = max(score, 0.8)
        elif re.search(rf"\b{re.escape(ref)}\b", text_lower):
            score = max(score, 0.35)

    return score


def _filename_score(query: str, filename: Optional[str]) -> float:
    if not filename:
        return 0.0
    return _keyword_score(query, filename)


def _and_score_boost(query: str, text: str) -> float:
    """Return an additive boost reflecting how many AND-terms hit ``text``.

    * All terms present  → :data:`AND_FULL_HIT_BOOST`.
    * Some terms present → proportional fraction of
      :data:`AND_PARTIAL_HIT_BOOST`.
    * No AND syntax used → ``0.0``.
    """
    terms = parse_and_terms(query)
    if not terms:
        return 0.0
    hits = _and_term_hits(text, terms)
    if hits == 0:
        return 0.0
    if hits == len(terms):
        return AND_FULL_HIT_BOOST
    return AND_PARTIAL_HIT_BOOST * (hits / len(terms))


def hybrid_score(
    *,
    query: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    lexical_score: float = 0.0,
    vector_score: float = 0.0,
    bm25_score_val: float = 0.0,
    query_word_count: int = 0,
) -> float:
    """Compute a hybrid relevance score with **adaptive weights**.

    Short queries (1–3 words) get heavier **vector** weight because
    semantic similarity is the strongest signal when there are few
    keywords.  Longer queries shift weight toward **lexical / BM25**
    because more keywords = more discriminating lexical signal.
    """
    metadata = metadata or {}
    keyword = _keyword_score(query, text)
    exact = _exact_phrase_score(query, text)
    section = _section_boost(query, text, metadata)
    filename = _filename_score(query, metadata.get("filename"))
    section_label = metadata.get("section_label")
    heading_bonus = 0.08 if section_label else 0.0
    and_boost = _and_score_boost(query, text)

    # Use BM25 if available, otherwise fall back to keyword ratio
    best_keyword = max(keyword, bm25_score_val)

    # ── Adaptive weight selection ────────────────────────────────────
    wc = query_word_count if query_word_count > 0 else len(
        [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
    )

    if wc <= 3:
        # Short queries: semantic embedding is the strongest signal
        w_vector  = 0.45
        w_lexical = 0.15
        w_keyword = 0.15
        w_section = 0.12
        w_exact   = 0.08
        w_fname   = 0.05
    else:
        # Longer queries: more keyword signal available
        w_vector  = 0.30
        w_lexical = 0.25
        w_keyword = 0.18
        w_section = 0.12
        w_exact   = 0.10
        w_fname   = 0.05

    score = (
        w_lexical * max(lexical_score, 0.0)
        + w_vector  * max(vector_score, 0.0)
        + w_keyword * best_keyword
        + w_section * section
        + w_exact   * exact
        + w_fname   * filename
        + heading_bonus
        + and_boost
    )
    return float(score)


def search_chunks(
    query: str, chunks: Sequence[SearchChunk], top_k: int = 5
) -> List[SearchMatch]:
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    lexical_scores: List[float]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words=None, max_features=50000, ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_vector, matrix).flatten().tolist()
    except Exception:
        lexical_scores = [_keyword_score(query, text) for text in texts]

    # Compute average doc length for BM25 normalisation
    avg_doc_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)

    scored: List[SearchMatch] = []
    for idx, chunk in enumerate(chunks):
        score = hybrid_score(
            query=query,
            text=chunk.text,
            metadata=chunk.metadata,
            lexical_score=lexical_scores[idx],
            bm25_score_val=_bm25_score(query, chunk.text, avg_doc_len),
        )
        scored.append(SearchMatch(score=score, chunk=chunk))

    scored.sort(key=lambda match: match.score, reverse=True)
    return scored[: max(top_k, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion — production-standard rank aggregation
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *rankings: Sequence[int],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Combine multiple ranking lists into a single fused ranking.

    Each ``ranking`` is a list of chunk indices sorted by relevance
    (best first).  The function returns ``(chunk_index, rrf_score)``
    pairs sorted by descending fused score.

    This is the standard technique used by Elasticsearch, Vespa, and
    Pinecone to combine vector + lexical rankings.

    Parameters
    ----------
    rankings:
        One or more lists of chunk indices, each sorted best-first.
    k:
        Smoothing constant (default 60, the standard in the literature).
    """
    scores: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_idx in enumerate(ranking):
            scores[chunk_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Re-rank scoring — deeper analysis for the second stage
# ─────────────────────────────────────────────────────────────────────────────

def compute_rerank_score(
    *,
    query_terms: List[str],
    text: str,
    vector_score: float,
    bm25: float,
    lexical: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute a deeper relevance score for the re-ranking stage.

    This is more expensive than :func:`hybrid_score` and is only run on
    the top-N candidates from stage 1.  It adds:

    * **Bigram overlap** — consecutive query-term pairs in the text
    * **Coverage ratio** — fraction of query terms that appear in text
    * **Term proximity** — are query terms near each other in the text?
    """
    metadata = metadata or {}
    text_lower = text.lower()
    text_tokens = re.findall(r"\w+", text_lower)

    if not query_terms or not text_tokens:
        return max(vector_score, 0.0)

    # ── Coverage: what fraction of query terms appear? ───────────────
    hits = [t for t in query_terms if t in text_lower]
    coverage = len(hits) / max(1, len(query_terms))

    # ── Bigram overlap ──────────────────────────────────────────────
    bigram_hits = 0
    if len(query_terms) >= 2:
        for i in range(len(query_terms) - 1):
            bigram = f"{query_terms[i]} {query_terms[i + 1]}"
            if bigram in text_lower:
                bigram_hits += 1
        bigram_score = bigram_hits / max(1, len(query_terms) - 1)
    else:
        bigram_score = 0.0

    # ── Term proximity: minimum window containing all hit terms ─────
    proximity_score = 0.0
    if len(hits) >= 2:
        positions: Dict[str, List[int]] = defaultdict(list)
        for idx, tok in enumerate(text_tokens):
            if tok in hits:
                positions[tok].append(idx)
        if len(positions) >= 2:
            # Find the smallest window containing at least one of each hit term
            all_positions = []
            for term, poses in positions.items():
                for p in poses:
                    all_positions.append((p, term))
            all_positions.sort()
            if all_positions:
                min_window = len(text_tokens)
                terms_in_window: Dict[str, int] = defaultdict(int)
                left = 0
                unique_hit_terms = set(hits)
                for right_idx in range(len(all_positions)):
                    terms_in_window[all_positions[right_idx][1]] += 1
                    while len(terms_in_window) >= len(unique_hit_terms):
                        window_size = all_positions[right_idx][0] - all_positions[left][0] + 1
                        min_window = min(min_window, window_size)
                        terms_in_window[all_positions[left][1]] -= 1
                        if terms_in_window[all_positions[left][1]] == 0:
                            del terms_in_window[all_positions[left][1]]
                        left += 1
                # Normalise: smaller window = higher score
                proximity_score = max(0.0, 1.0 - min_window / max(50.0, len(text_tokens)))

    # ── Section match bonus ─────────────────────────────────────────
    section_label = (metadata.get("section_label") or "").lower()
    section_hit = any(t in section_label for t in query_terms)
    section_bonus = 0.15 if section_hit else 0.0

    # ── Final re-rank score ─────────────────────────────────────────
    rerank = (
        0.30 * max(vector_score, 0.0)
        + 0.20 * bm25
        + 0.10 * lexical
        + 0.20 * coverage
        + 0.10 * bigram_score
        + 0.10 * proximity_score
        + section_bonus
    )
    return float(rerank)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence classification — multi-signal analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    scores: Sequence[float],
    section_refs: Sequence[str],
    exact_section_match: bool,
) -> str:
    """Classify retrieval confidence using multiple signals.

    Unlike the previous static-threshold approach, this uses:
    * **Score gap analysis** — a clear winner (#1 >> #2) boosts confidence
    * **Absolute score thresholds** — but lower than before, since the new
      scoring formula produces more discriminating scores
    * **Section match override** — exact section hits are always high
    """
    if not scores:
        return "low"

    top = scores[0]
    gap = scores[0] - scores[1] if len(scores) > 1 else 0.0

    if exact_section_match:
        return "high"
    if top >= 0.50:
        return "high"
    if top >= 0.38 and gap >= 0.08:
        return "high"
    if top >= 0.30 and gap >= 0.04:
        return "medium"
    if top >= 0.22:
        return "medium"
    return "low"
