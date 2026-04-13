"""
indexer.py
----------
scraped_data.json'daki yorumları Gemini text-embedding-004 ile embed eder
ve ChromaDB'ye kaydeder (persistent). Yeni mekanlar eklenince tekrar
çalıştırılabilir — zaten indexlenmiş mekanları atlar.

Kullanım:
    python indexer.py
    python indexer.py --input baska_dosya.json
    python indexer.py --reset   # DB'yi sıfırla ve baştan indexle
"""

import json
import os
import sys
import time
import logging
import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv
from google import genai
import chromadb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "gemini-embedding-001"
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "place_reviews"
BATCH_SIZE       = 50    # Gemini embedding batch boyutu
RATE_LIMIT_DELAY = 0.3   # batch'ler arası bekleme (saniye)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_place_name(entry: dict) -> str:
    if entry.get("place_name"):
        return entry["place_name"]
    url = entry.get("url", "")
    match = re.search(r"/place/([^/]+)", url)
    if match:
        return unquote(match.group(1).replace("+", " "))
    return "Bilinmeyen Mekan"


def place_slug(place_name: str) -> str:
    """Yer adından sabit 10 karakterlik bir ID prefix'i üretir."""
    return hashlib.md5(place_name.encode()).hexdigest()[:10]


def review_doc_id(place_name: str, idx: int) -> str:
    return f"{place_slug(place_name)}_{idx}"


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = client.models.embed_content(model=EMBED_MODEL, contents=text)
        embeddings.append(result.embeddings[0].values)
    return embeddings


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def index_place(
    collection,
    gemini_client: genai.Client,
    place_name: str,
    source_url: str,
    reviews: list[str],
) -> int:
    """Bir mekanın yorumlarını embed edip ChromaDB'ye ekler. Eklenen yorum sayısını döner."""

    # Zaten indexliyse atla
    existing = collection.get(where={"place_name": place_name}, limit=1)
    if existing["ids"]:
        log.info("  Zaten indexli, atlanıyor.")
        return 0

    added = 0
    for batch_start in range(0, len(reviews), BATCH_SIZE):
        batch_texts = reviews[batch_start : batch_start + BATCH_SIZE]
        embeddings  = embed_batch(gemini_client, batch_texts)

        collection.add(
            ids=[review_doc_id(place_name, batch_start + i) for i in range(len(batch_texts))],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[{"place_name": place_name, "source_url": source_url}] * len(batch_texts),
        )
        added += len(batch_texts)
        log.info(f"  {batch_start}–{batch_start + len(batch_texts) - 1}. yorumlar indexlendi.")

        if batch_start + BATCH_SIZE < len(reviews):
            time.sleep(RATE_LIMIT_DELAY)

    return added


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, reset: bool = False):
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY bulunamadı.")
        sys.exit(1)

    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        scraped_data = json.load(f)

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        log.warning("--reset: koleksiyon siliniyor ve yeniden oluşturuluyor.")
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total   = len(scraped_data)
    indexed = 0
    skipped = 0

    for i, entry in enumerate(scraped_data, start=1):
        place_name = extract_place_name(entry)
        reviews    = entry.get("reviews", [])
        source_url = entry.get("url", "")

        log.info(f"[{i}/{total}] '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning("  Yorum yok, atlanıyor.")
            skipped += 1
            continue

        added = index_place(collection, gemini_client, place_name, source_url, reviews)
        if added:
            indexed += 1
            log.info(f"  ✓ {added} yorum eklendi.")
        else:
            skipped += 1

    log.info(
        f"\nTamamlandı — Yeni: {indexed}, Atlanan (zaten var): {skipped} | "
        f"Koleksiyon toplam: {collection.count()} yorum"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB review indexer")
    parser.add_argument("--input", default="scraped_data.json")
    parser.add_argument("--reset", action="store_true", help="Mevcut DB'yi sıfırla")
    args = parser.parse_args()
    run(args.input, reset=args.reset)