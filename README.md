# ⚖️ LexBridge — U.S. Employment Law RAG System

> **A containerized Retrieval-Augmented Generation system for querying U.S. employment law.**  
> Built with FastAPI, Qdrant, MongoDB, and Groq/Gemini/Ollama — deployable with a single `docker-compose up`.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Development](#local-development)
- [API Reference](#api-reference)
- [Data Pipeline](#data-pipeline)
- [LLM Factory Pattern (Bonus)](#llm-factory-pattern-bonus)
- [Groq Integration](#groq-integration)
- [Arabic Support (Bonus)](#arabic-support-bonus)
- [Arabic Translation Pipeline](#arabic-translation-pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

**LexBridge** ingests raw, unstructured U.S. employment law pages (HTML scraped from Cornell LII, OSHA, and government sources), processes them through a custom NLP pipeline, and enables natural-language querying via a RAG architecture.

### Research Question

> *"Can a RAG system accurately retrieve and synthesize answers from raw legal web pages (containing messy HTML, boilerplate navigation, and inconsistent formatting) with sufficient precision to be useful for employment law queries?"*

### Key Features

- 🔍 **RAG Pipeline** — Query → Embed → Vector Search (language-filtered) → Context Injection → LLM Generation
- 📄 **Multi-format Ingestion** — PDF, DOCX, TXT, and HTML documents
- 🏭 **LLM Factory Pattern** — Switch between Groq, Gemini, OpenAI, or Ollama with one config change
- 🌍 **Arabic Support** — Text normalization, RTL-aware extraction, bilingual prompts, automatic translation pipeline
- 🐳 **Fully Containerized** — `docker-compose up` runs everything
- 📊 **Evaluation Framework** — 7 test cases with hallucination detection

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐  │
│  │  Streamlit  │───▶│   FastAPI (RAG)   │───▶│   MongoDB    |  │
│  │  Frontend   │     │                  │     │   (metadata) │  │
│  │  :8501      │     │  ┌────────────┐  │     └──────────────┘  │ 
│  └─────────────┘     │  │ Controllers│  │                       │
│                      │  │  ┌───────┐ │  │    ┌──────────────┐   │
│                      │  │  │ Data  │ │  │───▶│   Qdrant    │    │
│                      │  │  │ RAG   │ │  │    │  (vectors)   │   │
│                      │  │  └───────┘ │  │    └──────────────┘   │
│                      │  └────────────┘  │                       │
│                      │  ┌────────────┐  │     ┌──────────────┐  │
│                      │  │ LLM Factory│  │───▶│ Gemini/Ollama │  │
│                      │  │  ┌───────┐ │  │     │  (external)  │  │
│                      │  │  │OpenAI │ │  │     └──────────────┘  |
│                      │  │  │Ollama │ │  │                       │
│                      │  │  └───────┘ │  │                       │
│                      │  └────────────┘  │                       │
│                      │      :8000       │                       │
│                      └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Docker)

### Prerequisites
- Docker & Docker Compose installed
- A Groq API key (free at [console.groq.com](https://console.groq.com)) OR a Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd LexBridge

# 2. Configure your API key
cp docker/.env.example docker/.env
# Edit docker/.env — set LLM_BACKEND and corresponding API key:
#   - For Groq: LLM_BACKEND=groq, GROQ_API_KEY
#   - For Gemini: LLM_BACKEND=openai, OPENAI_API_KEY

# 3. Launch everything
docker-compose up --build

# 4. Access the system
#    API:      http://localhost:8000
#    Docs:     http://localhost:8000/docs
#    Frontend: http://localhost:8501
```

### First-Time Data Ingestion

After the containers are running, you need to ingest the corpus:

```bash
# Option A: Upload via API
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@path/to/document.pdf" \
  -F "collection=lexbridge"

# Option B: Run the ingestion pipeline
docker exec lexbridge-api python /scripts/pipeline.py
```

---

## Local Development

```bash
# 1. Install dependencies
cd src
pip install -r requirements.txt

# 2. Start MongoDB
mongod --dbpath ./data/db

# 3. Configure .env
cp .env.example .env
# Edit .env with your API key

# 4. Scrape data (first time only)
cd ..
python scripts/scrape_us.py
python scripts/scrape_us_more.py

# 5. Run ingestion pipeline
python scripts/pipeline.py

# 6. Start the API
cd src
uvicorn main:app --reload --port 8000

# 7. Start the frontend (separate terminal)
cd frontend
streamlit run app.py
```

---

## API Reference

### `GET /` — Health Check
```json
// Response
{
  "status": "ok",
  "app": "LexBridge",
  "version": "1.0.0",
  "db_connected": true,
  "vector_db_ready": true
}
```

### `POST /api/v1/data/upload` — Upload & Index Document
```bash
# Request (multipart form)
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@document.pdf" \
  -F "collection=lexbridge"
```
```json
// Response
{
  "message": "File processed successfully",
  "chunks_created": 24,
  "collection": "lexbridge"
}
```

### `POST /api/v1/nlp/ask` — Query the RAG System
```json
// Request
{
  "text": "What is at-will employment?",
  "collection": "lexbridge",
  "k": 5,
  "language": "en"
}
```
```json
// Response
{
  "answer": "At-will employment is a doctrine that allows either the employer or the employee to terminate the employment relationship at any time, for any reason...",
  "sources": [
    {
      "chunk_id": "abc-123",
      "text": "At-will employment refers to...",
      "score": 0.8945,
      "source_name": "Cornell Law",
      "source_url": "https://www.law.cornell.edu/wex/at-will_employment",
      "topic": "at-will employment",
      "language": "en"
    }
  ],
  "collection": "lexbridge",
  "model": "qwen/qwen3-32b"
}
```

**Note**: The `language` parameter filters vector search results — Arabic queries return only Arabic chunks, English queries return only English chunks. This prevents multilingual embedding models from mixing languages in retrieved context.

### `GET /api/v1/data/collection/{name}` — Collection Info
```json
// Response
{
  "collection": "lexbridge",
  "exists": true,
  "points_count": 446,
  "dimension": 3072
}
```

---

## Data Pipeline

### Phase 1: Scraping
```
scripts/scrape_us.py      → 14 core sources (Cornell LII, OSHA)
scripts/scrape_us_more.py → 21 additional sources (statutes, civil rights)
                           → Total: 35 raw HTML documents
```

### Phase 2: Parsing & Cleaning
- HTML → BeautifulSoup strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`
- Only block-level elements with 40+ chars retained
- PDF → PyMuPDF text extraction (preserves RTL for Arabic)
- DOCX → python-docx paragraph + table extraction

### Phase 3: Chunking Strategy

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `chunk_size` | 512 chars | ~100–130 tokens. Legal texts contain dense, self-contained paragraphs. 512 is small enough for focused semantic signal, large enough for LLM context. |
| `chunk_overlap` | 51 chars (10%) | Ensures boundary sentences appear in full in at least one chunk. ~1 sentence of overlap. |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Semantic hierarchy: paragraph → line → sentence → word. Respects natural text boundaries. |

### Phase 4: Embedding & Storage
- **Model**: `gemini-embedding-001` (3072 dimensions)
- **Vector DB**: Qdrant (local, embedded mode — no separate server)
- **Metadata DB**: MongoDB (chunk text, source URLs, topics, language tag)
- **Language Filtering**: Vector search filtered by language field to prevent cross-language retrieval

---

## LLM Factory Pattern (Bonus)

The system implements the **GoF Factory Method Pattern** to decouple LLM selection from business logic:

```
stores/llm/
├── base.py            # Abstract interface (BaseLLMProvider)
├── factory.py         # LLMFactory.create(settings) → provider
└── providers/
    ├── openai_provider.py   # OpenAI / Gemini / vLLM / LM Studio
    ├── groq_provider.py     # Groq (qwen/llama models) — optional
    └── ollama_provider.py   # Local Ollama (free, offline)
```

### Switching Providers

```env
# In .env — just change one line:

# Option 1: Gemini 2.5 Flash (default)
LLM_BACKEND="openai"
OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"

# Option 2: Groq (higher rate limits — 500k TPD)
LLM_BACKEND="groq"
GROQ_API_KEY="gsk_..."
GROQ_GENERATE_MODEL="qwen/qwen3-32b"

# Option 3: OpenAI
LLM_BACKEND="openai"
OPENAI_API_BASE="https://api.openai.com/v1"

# Option 4: Ollama (fully local)
LLM_BACKEND="ollama"
OLLAMA_BASE_URL="http://localhost:11434"
```

**Groq Integration** (Optional): The Groq provider uses Groq for generation (qwen/llama models) and Gemini for embeddings. This provides higher rate limits (500k TPD) if needed.

---

## Groq Integration (Optional)

Groq can be used as an alternative LLM provider for higher rate limits (500k tokens/day with qwen/qwen3-32b) if needed. The `GroqProvider` uses a hybrid approach:

- **Generation**: Groq API (qwen/qwen3-32b, llama-3.3-70b-versatile, etc.)
- **Embeddings**: Gemini (gemini-embedding-001, 3072 dimensions) — Groq has no embedding models

This is useful for batch translation and bulk ingestion without hitting free-tier rate limits.

---

## Arabic Support (Bonus)

### Challenges Addressed

1. **Tashkeel (Diacritics)**: `فَتَحَ` → `فتح` — Removed for consistent embedding
2. **Hamza Normalization**: `إ أ آ` → `ا` — Unifies variant spellings
3. **Tatweel (Kashida)**: `كـتـاب` → `كتاب` — Strips decorative elongation
4. **RTL Text Extraction**: PyMuPDF handles RTL at the character level
5. **Bilingual Prompts**: Separate Arabic prompt template for generation

### Arabic Test Case

```json
{
  "query": "ما هي قوانين العمل في الولايات المتحدة؟",
  "language": "ar"
}
```

See `scripts/evaluate.py` test case `adv_02` for evaluation results.

---

## Arabic Translation Pipeline

The `scripts/translate_and_ingest_ar.py` pipeline automatically translates English documents to Arabic and ingests them with language tagging:

```bash
python scripts/translate_and_ingest_ar.py
```

### Features

- **Full-document translation** — Uses `max_tokens=4096` (vs default 512) for complete document output
- **Rate-limit resilience** — Exponential backoff retry logic for Groq API
- **Resumable ingestion** — Caches translations and skips completed documents on restart
- **Language tagging** — All chunks tagged with `language: "ar"` for filtered retrieval
- **Chunk consistency** — Same 512-char chunks, 10% overlap as English pipeline

### How It Works

1. Iterates English docs in `data/raw/us/`
2. For each document, translates to Arabic with `max_tokens=4096`
3. Caches translation to `data/raw/us/arabic/{key}_ar.txt`
4. Creates `.done` sentinel file to track completion
5. Chunks and embeds with Gemini (3072-dim)
6. Upserts to Qdrant with `language: "ar"` metadata
7. Persists to MongoDB

This enables bilingual querying — users can ask in Arabic and retrieve only Arabic context, preventing cross-language contamination.

---

## Evaluation

Run the evaluation suite:
```bash
python scripts/evaluate.py
```

### Test Cases

| ID | Category | Query | Expected |
|----|----------|-------|----------|
| std_01 | Standard | "What is at-will employment?" | Strong retrieval, keyword coverage |
| std_02 | Standard | "What does the FMLA require?" | FMLA-related chunks, "12 weeks" |
| std_03 | Standard | "Federal minimum wage under FLSA?" | Minimum wage, FLSA chunks |
| edge_01 | Edge Case | "What about work?" | Low semantic signal → poor retrieval |
| edge_02 | Edge Case | "Minimum wage in New York State?" | No state-level data → uncertainty |
| adv_01 | Adversarial | "Labor laws on Mars?" | Must refuse — no hallucination |
| adv_02 | Adversarial | Arabic query on English corpus | Embedding mismatch expected |

---

## Project Structure

```
LexBridge/
├── docker-compose.yml          # ← Run: docker-compose up
├── Dockerfile                  # API container
├── Dockerfile.frontend         # Streamlit container
├── .dockerignore
│
├── src/                        # FastAPI application (MVC)
│   ├── main.py                 # App factory + lifespan
│   ├── requirements.txt
│   ├── .env / .env.example
│   ├── controllers/            # Business logic (Controller)
│   │   └── __init__.py         #   DataController, RAGController
│   ├── models/                 # Data models (Model)
│   │   └── db_schemes/         #   Pydantic schemas
│   ├── routes/                 # API endpoints (View)
│   │   ├── __init__.py         #   Router definitions
│   │   └── schema/             #   Request/Response models
│   ├── stores/                 # External service integrations
│   │   ├── llm/
│   │   │   ├── base.py         #   Abstract LLM interface
│   │   │   ├── factory.py      #   LLM Factory
│   │   │   ├── providers/      #   OpenAI, Ollama
│   │   │   └── tempelate/      #   Prompt templates (en, ar)
│   │   └── vectordb/
│   │       └── provider/       #   Qdrant integration
│   └── helpers/
│       ├── config.py           #   Settings (pydantic-settings)
│       └── arabic.py           #   Arabic NLP normalization
│
├── scripts/                    # Data pipeline scripts
│   ├── scrape_us.py            #   Web scraper (14 sources)
│   ├── scrape_us_more.py       #   Extended scraper (21 sources)
│   ├── pipeline.py             #   Ingestion pipeline
│   ├── translate_and_ingest_ar.py #   Arabic translation pipeline
│   └── evaluate.py             #   RAG evaluation (7 test cases)
│
├── data/raw/us/                # Scraped HTML corpus (35 docs)
├── frontend/                   # Streamlit UI
│   ├── app.py
│   └── requirements.txt
│
├── docker/                     # Docker env files
│   └── .env                    #   Environment configuration
│
├── TECHNICAL_REPORT.md         # Technical documentation
└── README.md                   # ← You are here
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.11, FastAPI |
| **LLM** | Gemini 2.5 Flash (default), Groq/Ollama (optional) |
| **Embeddings** | Gemini Embedding 001 (3072-dim) |
| **Vector DB** | Qdrant (embedded mode) |
| **Metadata DB** | MongoDB 7.0 |
| **Document Parsing** | PyMuPDF, python-docx, BeautifulSoup |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Frontend** | Streamlit |
| **Containerization** | Docker, Docker Compose |

---

## License

This project was built as a final engineering project for an NLP course.
