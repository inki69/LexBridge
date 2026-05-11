"""
Translate the existing English legal corpus to Arabic and ingest into Qdrant.
Run from the project root: python scripts/translate_and_ingest_ar.py

Reads:   data/raw/us/html/*.html  (via scrape_metadata.json)
Caches:  data/raw/us/arabic/*.txt (skipped on re-runs)
Ingests: into the 'lexbridge' collection with language='ar'

Arabic normalization (tashkeel removal, hamza, tatweel, etc.) is applied
automatically by DataController because is_arabic() detects the translated text.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "us"
METADATA_FILE = DATA_DIR / "scrape_metadata.json"
HTML_DIR = DATA_DIR / "html"
AR_DIR = DATA_DIR / "arabic"
COLLECTION = "lexbridge"

os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.factory import LLMFactory
from stores.vectordb.provider import QdrantProvider
from controllers import DataController

_TRANSLATION_PROMPT = """\
Translate the following U.S. employment law text into Modern Standard Arabic (فصحى).
Preserve all legal terms accurately. Output only the Arabic translation — no explanations, no English, no preamble.

Text:
{text}"""


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = []
    for el in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text) > 40:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


_TRANSLATION_MAX_TOKENS = 4096  # full documents need far more than the 512-token RAG answer limit

def _translate_with_retry(llm, text: str, retries: int = 5, base_delay: float = 10.0) -> str:
    for attempt in range(retries):
        try:
            return llm.generate(_TRANSLATION_PROMPT.format(text=text), max_tokens=_TRANSLATION_MAX_TOKENS)
        except Exception as exc:
            # Catch OpenAI RateLimitError by name so we don't need to import openai here
            if "RateLimitError" in type(exc).__name__:
                if attempt == retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)
                print(f"    [rate limit] waiting {wait:.0f}s (retry {attempt + 1}/{retries - 1})...")
                time.sleep(wait)
            else:
                raise


async def run():
    AR_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_FILE.exists():
        print(f"[ERROR] Metadata file not found: {METADATA_FILE}")
        print("Run scripts/scrape_us.py first.")
        sys.exit(1)

    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    settings = get_settings()

    mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = mongo_client[settings.MONGODB_DB_NAME]

    vector_store = QdrantProvider(
        path=settings.VECTOR_DB_PATH,
        distance_metric=settings.VECTOR_DISTANCE_METRIC,
    )

    llm = LLMFactory.create(settings)
    controller = DataController(db=db, vector_store=vector_store, llm=llm)

    print(f"[INFO] {len(metadata)} documents — translating to Arabic & ingesting")
    print("=" * 60)

    total_chunks = 0

    for entry in metadata:
        key = entry["key"]
        html_path = HTML_DIR / entry["file"]
        ar_cache = AR_DIR / f"{key}_ar.txt"
        done_flag = AR_DIR / f"{key}_ar.done"

        # ── Skip if already fully translated + ingested ───────────────────
        if done_flag.exists():
            print(f"  [DONE] {key} — already ingested, skipping")
            # Count cached chunks toward total for the summary
            continue

        if not html_path.exists():
            print(f"  [SKIP] {key} — HTML not found")
            continue

        # ── Translation (use cache if available) ──────────────────────────
        if ar_cache.exists():
            arabic_text = ar_cache.read_text(encoding="utf-8")
            print(f"  [CACHE] {key} — translation cached, ingesting...")
        else:
            html = html_path.read_text(encoding="utf-8")
            clean_en = extract_text_from_html(html)

            if len(clean_en.strip()) < 100:
                print(f"  [SKIP] {key} — insufficient text after cleaning")
                continue

            print(f"  [TRANSLATE] {key} ({len(clean_en)} chars)...", end="", flush=True)
            arabic_text = await asyncio.to_thread(_translate_with_retry, llm, clean_en)
            ar_cache.write_text(arabic_text, encoding="utf-8")
            print(f" -> {len(arabic_text)} chars")

            # Small pause between translation API calls to stay within rate limits
            await asyncio.sleep(2)

        # ── Ingest (chunk → embed → store) ────────────────────────────────
        doc_meta = {
            "url": entry.get("url", ""),
            "source": entry.get("source", ""),
            "topic": entry.get("topic", ""),
            "language": "ar",
        }

        chunks_created = await controller.process_file(
            file_content=arabic_text.encode("utf-8"),
            filename=f"{key}_ar.txt",
            collection=COLLECTION,
            chunk_size=settings.FILE_CHUNK_SIZE,
            metadata=doc_meta,
        )

        # Mark as fully done so re-runs skip this document entirely
        done_flag.touch()

        total_chunks += chunks_created
        print(f"  [OK] {key}_ar -> {chunks_created} chunks")

    print("=" * 60)
    print(f"[DONE] Arabic chunks ingested: {total_chunks}")
    print(f"       Translations cached in: {AR_DIR}")
    mongo_client.close()


if __name__ == "__main__":
    asyncio.run(run())
