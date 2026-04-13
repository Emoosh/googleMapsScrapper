"""
indexer.py
----------
scraped_data.json'daki yorumları intfloat/multilingual-e5-large ile embed eder
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
import logging
import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote

from sentence_transformers import SentenceTransformer
import torch
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
EMBED_MODEL     = "intfloat/multilingual-e5-large"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "place_reviews"

# CUDA > MPS > CPU otomatik seçim
if torch.cuda.is_available():
    DEVICE = "cuda"
    BATCH_SIZE = 256
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    BATCH_SIZE = 64
else:
    DEVICE = "cpu"
    BATCH_SIZE = 32


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
    return hashlib.md5(place_name.encode()).hexdigest()[:10]


def review_doc_id(place_name: str, idx: int) -> str:
    return f"{place_slug(place_name)}_{idx}"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def index_place(
    collection,
    model: SentenceTransformer,
    place_name: str,
    source_url: str,
    reviews: list[str],
) -> int:
    """Bir mekanın yorumlarını embed edip ChromaDB'ye ekler. Eklenen yorum sayısını döner."""

    existing = collection.get(where={"place_name": place_name}, limit=1)
    if existing["ids"]:
        log.info("  Zaten indexli, atlanıyor.")
        return 0

    # multilingual-e5 için "passage: " prefix'i gerekli
    prefixed = [f"passage: {r}" for r in reviews]
    embeddings = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=[review_doc_id(place_name, i) for i in range(len(reviews))],
        embeddings=embeddings,
        documents=reviews,
        metadatas=[{"place_name": place_name, "source_url": source_url}] * len(reviews),
    )

    return len(reviews)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, reset: bool = False):
    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)

    log.info(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}")
    log.info(f"Model yükleniyor: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    with open(input_file, encoding="utf-8") as f:
        scraped_data = json.load(f)

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

        added = index_place(collection, model, place_name, source_url, reviews)
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