# AI Doc Backend — Multi-Engine OCR, Bilingual RAG & Full-Duplex Voice AI

[![Architected & Built by Mohamed Anwar](https://img.shields.io/badge/Architected_%26_Built_by-Mohamed_Anwar-4f46e5?style=for-the-badge&logo=github)](https://github.com/MohamedAnwar070)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **Solely architected, engineered, and built end-to-end by [Mohamed Anwar](https://github.com/MohamedAnwar070)** — spanning the multi-engine OCR & RAG backend, the interactive React web workspace, and the GPU-accelerated native Android client.

---

## 🌐 The AI Doc Ecosystem

| Repository | Description | Tech Stack |
| :--- | :--- | :--- |
| **[AI-Doc-Backend](https://github.com/MohamedAnwar070/AI-Doc-Backend)** *(Current)* | Multi-engine OCR pipeline, hybrid semantic RAG, BGE-M3 embeddings, and real-time WebSocket Voice AI | FastAPI, Python, MongoDB, PaddleOCR-VL-1.5, Whisper, XTTS-v2, Silero VAD |
| **[AI-Doc-Frontend](https://github.com/MohamedAnwar070/AI-Doc-Frontend)** | Interactive web platform with side-by-side OCR visual editor, PDF/Excel synchronization, and voice assistant | React 18, TypeScript, Vite, Tailwind CSS, TipTap, shadcn/ui |
| **[AI-Doc-Android](https://github.com/MohamedAnwar070/AI-Doc-Android)** | Native Android client with real-time 3D Filament GPU shader VoiceOrb, RAG chat, and document viewer | Kotlin, Jetpack Compose, Material 3, Google Filament 3D, OkHttp |

---

## ✨ Key Engineering Highlights

### 1. Multi-Engine OCR & Native Document Pipeline
- **Pluggable OCR Factory**: Seamlessly switch between **PaddleOCR-VL-1.5** (Vision-Language Model), **Surya OCR**, **olmOCR (7B VLM)**, **Google Cloud Document AI / Vision AI**, **OpenAI Vision OCR**, **PaddleOCR Classic**, and **Tesseract**.
- **Zero-Cost Native Extraction**: Automatically detects digital text-layer PDFs (`pdfplumber`), Word documents (`.docx`), and Excel spreadsheets (`.xlsx`) to extract structured pages and HTML tables without unnecessary OCR overhead.
- **Image Preprocessing & Layout Recovery**: Automated deskewing, Sauvola binarization, border cropping, CLAHE contrast enhancement, bounding-box reading-order reconstruction, and HTML table synthesis.
- **Bilingual Tamil & English Normalization**: NFC Unicode normalization, Tamil glyph confusable correction, and per-page translation with hash-verified caching.

### 2. Production RAG & Hybrid Semantic Search
- **Parent–Child Hierarchical Chunking**: Links `800`-character child search chunks to `2,400`-character parent context windows so the LLM receives complete legal/document clauses without truncation.
- **Deep Metadata Enrichment**: Automatically extracts document titles, classifications, bilingual language tags (`en`, `ta`, `bilingual`), dates, parties, and section hierarchies onto every indexed chunk.
- **2-Tier Vector Caching & BGE-M3 Embeddings**: Supports `1024-d` **BGE-M3** embeddings (in-process or remote microservice) with content-addressed MongoDB vector caching to eliminate redundant re-indexing costs.
- **Shared Turn Pipeline**: Unified intent routing (`intent -> retrieve -> persona -> answer`) with boolean `AND` (`+`) query promotion and automated LLM refusal recovery when document evidence is present.

### 3. Real-Time Full-Duplex Voice AI (WebSocket)
- **Silero VAD v4 Gate**: Sub-second end-of-speech detection over streaming PCM audio frames.
- **Faster-Whisper STT**: High-accuracy multilingual & Tanglish speech-to-text transcription (`int8` CPU / `float16` GPU).
- **Coqui XTTS-v2 & Piper TTS**: Low-latency multilingual speech synthesis with custom speaker voice cloning support.

### 4. Resilient Job Orchestration & Security
- **Durable Queue & Crash Recovery**: Incremental page-by-page persistence (`status.json`, `result.txt`, `pages/page_XXXX.json`) allows interrupted 300+ page jobs to resume from the exact last completed page.
- **JWT Auth & Role-Based Access**: Secure user/admin authentication, hierarchical category trees (up to 5 levels), custom document tags, and per-user saved chat histories.

---

## 🏗️ System Architecture

```text
ai-doc-retrieval-backend/
├── api/                        # FastAPI routers
│   ├── auth.py                 # JWT login, registration & role guards
│   ├── upload.py               # Resilient PDF / DOCX / XLSX upload endpoints
│   ├── status.py               # Job progress, stats, search & filtering
│   ├── result.py               # Text, structured JSON, layout & HTML result streaming
│   ├── semantic_search.py      # Hybrid RAG & search suggestion endpoints
│   ├── chats.py                # Persistent document chat sessions
│   ├── categories.py           # Nested category tree management
│   ├── tags.py                 # Document tagging endpoints
│   └── translation.py          # Page & batch bilingual translation
├── core/                       # Core RAG, OCR & worker orchestration
│   ├── turn_pipeline.py        # Unified RAG turn pipeline (Chat + Voice)
│   ├── retrieval.py            # Hybrid vector + lexical + section retrieval
│   ├── semantic_search.py      # Parent-child chunking & vector indexing
│   ├── embeddings.py           # BGE-M3 embedding client & 2-tier cache
│   ├── llm.py                  # OpenAI / Local / Remote GGUF LLM router
│   ├── worker.py               # Multiprocessing PDF OCR worker
│   ├── word_worker.py          # Native DOCX extraction worker
│   └── excel_worker.py         # Native XLSX sheet extraction worker
├── ocr/                        # Pluggable OCR engine implementations
│   ├── factory.py              # Dynamic OCR engine factory
│   ├── paddle_vl_engine.py     # PaddleOCR-VL-1.5 local VLM engine
│   ├── paddle_vl_remote_engine.py # PaddleOCR-VL-1.5 remote microservice client
│   ├── surya_engine.py         # Surya OCR layout-aware engine
│   ├── olmocr_engine.py        # AllenAI olmOCR-2-7B engine
│   ├── google_engine.py        # Google Cloud Document AI engine
│   └── _table_reconstructor.py # Spatial bounding-box to HTML table builder
├── voice/                      # Real-time WebSocket voice conversation stack
│   ├── ws_handler.py           # Full-duplex WebSocket session handler
│   ├── conversation_engine.py  # VAD -> STT -> RAG -> TTS turn orchestrator
│   ├── stt_service.py          # Faster-Whisper STT client/fallback
│   └── tts_service.py          # XTTS-v2 / Kokoro / Piper TTS synthesizer
├── services/                   # Standalone deployable AI microservices
│   ├── embedding/              # Dedicated BGE-M3 embedding FastAPI server
│   ├── paddle-vl/              # Dedicated PaddleOCR-VL-1.5 inference server
│   └── voice/                  # Dedicated Whisper + XTTS-v2 + Silero VAD server
├── db/                         # Async Motor MongoDB repositories & models
├── storage/                    # Local & Supabase storage + crash resume handlers
├── config.py                   # Strongly-typed Pydantic Settings configuration
└── main.py                     # Application entrypoint & lifespan manager
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python** `3.11+`
- **MongoDB** `6.0+` (running locally on `mongodb://localhost:27017` or remote URI)
- Optional: LibreOffice (`soffice`) for DOCX/XLSX-to-PDF source preview rendering

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/MohamedAnwar070/AI-Doc-Backend.git
cd AI-Doc-Backend

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Run the Backend Server

```bash
# Development mode
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Once running, explore the interactive OpenAPI documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

## 📡 Core API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register a new user account |
| `POST` | `/api/auth/login` | Authenticate and receive a JWT bearer token |
| `POST` | `/api/upload` | Upload PDF or Word document for OCR & RAG indexing |
| `POST` | `/api/upload/excel` | Upload Excel workbook and process selected sheet |
| `GET` | `/api/jobs` | Paginated list of jobs with status, category, and tag filters |
| `GET` | `/api/status/{job_id}` | Real-time processing progress and indexing health |
| `GET` | `/api/result/{job_id}` | Extracted text, page count, and document metadata |
| `PATCH` | `/api/result/{job_id}` | Save edited OCR text / structured visual layout blocks |
| `POST` | `/api/result/{job_id}/translate/page/{page}` | Translate a specific page between English (`en`) and Tamil (`ta`) |
| `POST` | `/api/semantic-search` | Execute hybrid semantic search & RAG answer generation |
| `GET` | `/api/chats` | List saved document chat sessions |
| `WS` | `/api/voice/ws` | Full-duplex real-time voice conversation WebSocket |

---

## 🧪 Running Tests

```bash
pytest -q
```

---

## 👨‍💻 Author & Architect

**Mohamed Anwar**  
*Sole Creator & Full-Stack AI / Mobile / Backend Engineer*

Designed, architected, and implemented from scratch by **Mohamed Anwar** as a unified, production-grade Document Intelligence & Voice RAG ecosystem across Backend, Web, and Android.

- **GitHub**: [@MohamedAnwar070](https://github.com/MohamedAnwar070)
- **Repositories**:
  - [AI-Doc-Backend](https://github.com/MohamedAnwar070/AI-Doc-Backend)
  - [AI-Doc-Frontend](https://github.com/MohamedAnwar070/AI-Doc-Frontend)
  - [AI-Doc-Android](https://github.com/MohamedAnwar070/AI-Doc-Android)

---

## 📝 License

This project is licensed under the MIT License.
