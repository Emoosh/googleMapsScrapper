"""
Local embedding + ChromaDB yükleyici.

Kullanım:
  pip install sentence-transformers psycopg2-binary chromadb
  python local_indexer.py --db-url postgresql://... --chroma-host <railway-host> --chroma-port 8000

Çevre değişkenleri ile de çalışır:
  DATABASE_URL=... CHROMA_HOST=... CHROMA_PORT=... python local_indexer.py
"""

import argparse
import logging
import os
import sys

import chromadb
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

COLLECTION_NAME = "places"
MODEL_NAME      = "intfloat/multilingual-e5-large"
BATCH_SIZE      = 32


def _fetch_places(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.id,
                p.source_url,
                p.name,
                p.address,
                p.rating,
                p.total_ratings,
                COALESCE(
                    array_to_string(ARRAY(
                        SELECT content FROM reviews WHERE place_id = p.id
                    ), ' | '),
                    ''
                ) AS reviews_text
            FROM places p
            ORDER BY p.id
        """)
        return [dict(row) for row in cur.fetchall()]


def _build_document(place: dict) -> str:
    parts = [f"Mekan: {place['name'] or ''}"]
    if place.get("address"):
        parts.append(f"Adres: {place['address']}")
    if place.get("rating"):
        parts.append(f"Puan: {place['rating']}")
    if place.get("reviews_text"):
        parts.append(f"Yorumlar: {place['reviews_text'][:2000]}")
    return " | ".join(parts)


def run(db_url: str, chroma_host: str, chroma_port: int, reset: bool = False):
    log.info(f"PostgreSQL'e bağlanılıyor: {db_url[:40]}...")
    conn = psycopg2.connect(db_url)

    log.info(f"ChromaDB'ye bağlanılıyor: {chroma_host}:{chroma_port}")
    client     = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if reset:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Collection sıfırlandı.")

    log.info(f"Model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    places = _fetch_places(conn)
    conn.close()
    log.info(f"{len(places)} mekan bulundu.")

    existing_ids = set(collection.get(ids=[str(p["id"]) for p in places])["ids"])
    to_index     = [p for p in places if str(p["id"]) not in existing_ids]
    log.info(f"İndekslenmemiş: {len(to_index)}, zaten mevcut: {len(existing_ids)}")

    if not to_index:
        log.info("Tüm mekanlar zaten indekslenmiş.")
        return

    for i in range(0, len(to_index), BATCH_SIZE):
        batch = to_index[i:i + BATCH_SIZE]
        docs  = [_build_document(p) for p in batch]
        ids   = [str(p["id"]) for p in batch]
        metas = [
            {
                "name":   p["name"] or "",
                "url":    p["source_url"] or "",
                "rating": float(p["rating"] or 0),
            }
            for p in batch
        ]

        log.info(f"Embedding: {i+1}-{i+len(batch)}/{len(to_index)}")
        embeddings = model.encode(docs, show_progress_bar=False).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        log.info(f"  ✓ {len(batch)} mekan yüklendi.")

    log.info(f"Tamamlandı. Toplam collection boyutu: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",      default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    parser.add_argument("--reset",       action="store_true", help="Collection'ı sıfırdan oluştur")
    args = parser.parse_args()

    if not args.db_url:
        log.error("--db-url veya DATABASE_URL gerekli")
        sys.exit(1)

    run(args.db_url, args.chroma_host, args.chroma_port, args.reset)