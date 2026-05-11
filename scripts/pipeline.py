"""
Ingestion pipeline: reads scraped HTML → cleans → chunks → embeds → Qdrant + MongoDB.
Run from the project root: python scripts/pipeline.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "us"
METADATA_FILE = DATA_DIR / "scrape_metadata.json"
HTML_DIR = DATA_DIR / "html"
COLLECTION = "lexbridge"

# Switch to src/ so relative paths in config (.env, qdrant_data) resolve correctly
os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.factory import LLMFactory
from stores.vectordb.provider import QdrantProvider
from controllers import DataController


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Collect block-level text
    paragraphs = []
    for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]):
        text = element.get_text(separator=" ", strip=True)
        if len(text) > 40:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


async def run():
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

    # Use the LLM Factory Pattern (same as main.py)
    llm = LLMFactory.create(settings)

    controller = DataController(db=db, vector_store=vector_store, llm=llm)

    # Drop and rebuild chunks collection for a clean run
    await db.chunks.drop()
    print(f"[INFO] Dropped MongoDB chunks collection")

    print(f"[INFO] Processing {len(metadata)} documents into collection '{COLLECTION}'")
    print("=" * 60)

    total_chunks = 0
    for entry in metadata:
        html_path = HTML_DIR / entry["file"]
        if not html_path.exists():
            print(f"  [SKIP] {entry['key']} — file not found")
            continue

        html = html_path.read_text(encoding="utf-8")
        clean_text = extract_text_from_html(html)

        if len(clean_text.strip()) < 100:
            print(f"  [SKIP] {entry['key']} — insufficient text after cleaning")
            continue

        doc_meta = {
            "url": entry.get("url", ""),
            "source": entry.get("source", ""),
            "topic": entry.get("topic", ""),
            "language": entry.get("language", "en"),
        }

        chunks_created = await controller.process_file(
            file_content=clean_text.encode("utf-8"),
            filename=f"{entry['key']}.txt",
            collection=COLLECTION,
            chunk_size=settings.FILE_CHUNK_SIZE,
            metadata=doc_meta,
        )

        total_chunks += chunks_created
        print(f"  [OK] {entry['key']} -> {chunks_created} chunks")

    print("=" * 60)
    print(f"[DONE] Total chunks ingested: {total_chunks}")
    mongo_client.close()


if __name__ == "__main__":
    asyncio.run(run())
