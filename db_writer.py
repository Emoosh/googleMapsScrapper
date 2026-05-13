import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool
import redis
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:changeme@db:5432/neredenevar")

URLS_QUEUE     = "queue:urls:db"
RAW_QUEUE      = "queue:places:raw_db"
ANALYSIS_QUEUE = "queue:places:to_db"

_pool: psycopg2.pool.ThreadedConnectionPool = None


def _init_pool():
    global _pool
    _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, DATABASE_URL)
    log.info("DB connection pool oluşturuldu.")


def get_conn():
    return _pool.getconn()


def release_conn(conn):
    _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _migrate():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS place_urls (
                    id            SERIAL PRIMARY KEY,
                    url           TEXT UNIQUE,
                    status        TEXT DEFAULT 'pending',
                    discovered_at TIMESTAMPTZ DEFAULT NOW(),
                    scraped_at    TIMESTAMPTZ,
                    analyzed_at   TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS places (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT,
                    source_url    TEXT UNIQUE,
                    place_type    TEXT,
                    location      GEOMETRY(Point, 4326),
                    lat           FLOAT,
                    lng           FLOAT,
                    total_reviews INT,
                    address       TEXT,
                    phone         TEXT,
                    rating        FLOAT,
                    total_ratings INT,
                    website_url   TEXT,
                    website_type  TEXT,
                    images        TEXT[],
                    scraped_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id       SERIAL PRIMARY KEY,
                    place_id INT REFERENCES places(id) ON DELETE CASCADE,
                    content  TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS place_analysis (
                    id                      SERIAL PRIMARY KEY,
                    place_id                INT UNIQUE REFERENCES places(id) ON DELETE CASCADE,
                    analyzed_at             TIMESTAMPTZ DEFAULT NOW(),
                    overall_score           FLOAT,
                    summary                 TEXT,
                    ideal_for               TEXT,
                    price_level             TEXT,
                    score_service           FLOAT,
                    score_price_performance FLOAT,
                    score_atmosphere        FLOAT,
                    scores_extra            JSONB,
                    highlights              TEXT[],
                    downsides               TEXT[],
                    popular_items           TEXT[],
                    tags                    TEXT[],
                    wifi_priz               TEXT,
                    kalabalik_seviyesi      TEXT
                )
            """)
            # Backward-compat column additions
            for col in [
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS location GEOMETRY(Point, 4326)",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS lat FLOAT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS lng FLOAT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS address TEXT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS phone TEXT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS rating FLOAT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS total_ratings INT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS website_url TEXT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS website_type TEXT",
                "ALTER TABLE places ADD COLUMN IF NOT EXISTS images TEXT[]",
                "ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMPTZ DEFAULT NOW()",
                "ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS wifi_priz TEXT",
                "ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS kalabalik_seviyesi TEXT",
            ]:
                cur.execute(col)
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'place_analysis'::regclass
                        AND contype = 'u'
                        AND conname = 'place_analysis_place_id_key'
                    ) THEN
                        ALTER TABLE place_analysis ADD CONSTRAINT place_analysis_place_id_key UNIQUE (place_id);
                    END IF;
                END $$
            """)
        conn.commit()
        log.info("[migrate] Tablolar güncellendi.")
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Write: Phase 1 — URL kaydı
# ---------------------------------------------------------------------------

def write_url(payload: dict):
    url = payload.get("url", "")
    if not url:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO place_urls (url, status, discovered_at)
                VALUES (%s, 'pending', NOW())
                ON CONFLICT (url) DO NOTHING
            """, (url,))
        conn.commit()
        log.info(f"  ✓ URL kaydedildi: {url[:60]}")
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Write: Phase 2 — Ham scrape verisi (images dahil)
# ---------------------------------------------------------------------------

def write_raw_place(payload: dict):
    url  = payload.get("url", "")
    lat  = payload.get("lat")
    lng  = payload.get("lng")
    # lat/lng are GENERATED ALWAYS columns (computed from location) — cannot be inserted directly
    point_lng = float(lng) if lng is not None else 0.0
    point_lat = float(lat) if lat is not None else 0.0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO places
                    (name, source_url, place_type, location, total_reviews,
                     address, phone, rating, total_ratings, website_url, website_type,
                     images, scraped_at)
                VALUES
                    (%s, %s, 'cafe', ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s,
                     %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (source_url) DO UPDATE SET
                    name          = EXCLUDED.name,
                    location      = EXCLUDED.location,
                    total_reviews = EXCLUDED.total_reviews,
                    address       = EXCLUDED.address,
                    phone         = EXCLUDED.phone,
                    rating        = EXCLUDED.rating,
                    total_ratings = EXCLUDED.total_ratings,
                    website_url   = EXCLUDED.website_url,
                    website_type  = EXCLUDED.website_type,
                    images        = EXCLUDED.images,
                    scraped_at    = NOW()
                RETURNING id
            """, (
                payload.get("name") or "Bilinmeyen", url,
                point_lng, point_lat,
                payload.get("total_reviews_scraped") or len(payload.get("reviews", [])),
                payload.get("address"), payload.get("phone"),
                payload.get("rating"), payload.get("total_ratings"),
                payload.get("website_url"), payload.get("website_type"),
                payload.get("images") or [],
            ))

            row = cur.fetchone()
            if not row:
                conn.rollback()
                log.warning(f"place upsert sonuç döndürmedi: {url}")
                return
            place_id = row[0]

            cur.execute("DELETE FROM reviews WHERE place_id = %s", (place_id,))
            reviews = payload.get("reviews", [])
            if reviews:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO reviews (place_id, content) VALUES %s",
                    [(place_id, r) for r in reviews],
                )

            cur.execute("""
                UPDATE place_urls SET status='scraped', scraped_at=NOW() WHERE url=%s
            """, (url,))

        conn.commit()
        log.info(f"  ✓ Ham veri yazıldı: '{payload.get('name')}' (id={place_id}, {len(reviews)} yorum)")
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Write: Phase 3 — Analiz sonucu
# ---------------------------------------------------------------------------

def write_analysis(payload: dict):
    url      = payload.get("url", "")
    analysis = payload.get("analysis", {})
    if not analysis:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM places WHERE source_url = %s", (url,))
            row = cur.fetchone()
            if not row:
                log.warning(f"  ✗ place_analysis yazılamadı, places'da yok: {url[:60]}")
                return
            place_id = row[0]

            scores       = analysis.get("scores", {})
            base_keys    = {"hizmet", "fiyat_performans", "atmosfer"}
            scores_extra = {k: v for k, v in scores.items() if k not in base_keys}

            cur.execute("""
                INSERT INTO place_analysis (
                    place_id, analyzed_at, overall_score, summary, ideal_for, price_level,
                    score_service, score_price_performance, score_atmosphere,
                    scores_extra, highlights, downsides, popular_items, tags,
                    wifi_priz, kalabalik_seviyesi
                ) VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (place_id) DO UPDATE SET
                    analyzed_at             = NOW(),
                    overall_score           = EXCLUDED.overall_score,
                    summary                 = EXCLUDED.summary,
                    ideal_for               = EXCLUDED.ideal_for,
                    price_level             = EXCLUDED.price_level,
                    score_service           = EXCLUDED.score_service,
                    score_price_performance = EXCLUDED.score_price_performance,
                    score_atmosphere        = EXCLUDED.score_atmosphere,
                    scores_extra            = EXCLUDED.scores_extra,
                    highlights              = EXCLUDED.highlights,
                    downsides               = EXCLUDED.downsides,
                    popular_items           = EXCLUDED.popular_items,
                    tags                    = EXCLUDED.tags,
                    wifi_priz               = EXCLUDED.wifi_priz,
                    kalabalik_seviyesi      = EXCLUDED.kalabalik_seviyesi
            """, (
                place_id,
                analysis.get("genel_puan"),
                analysis.get("ozet"),
                analysis.get("kim_icin_ideal"),
                analysis.get("fiyat_seviyesi"),
                scores.get("hizmet"),
                scores.get("fiyat_performans"),
                scores.get("atmosfer"),
                json.dumps(scores_extra, ensure_ascii=False),
                analysis.get("one_cikanlar") or [],
                analysis.get("eksiler") or [],
                analysis.get("populer_urunler") or [],
                analysis.get("etiketler") or [],
                analysis.get("wifi_priz", "belirtilmemiş"),
                analysis.get("kalabalik_seviyesi", "belirtilmemiş"),
            ))

            cur.execute("""
                UPDATE place_urls SET status='analyzed', analyzed_at=NOW() WHERE url=%s
            """, (url,))

        conn.commit()
        log.info(f"  ✓ Analiz yazıldı: place_id={place_id}")
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

_worker_stats: dict = {"processed": 0, "failed": 0, "running": False}

_QUEUES = [URLS_QUEUE, RAW_QUEUE, ANALYSIS_QUEUE]

_HANDLERS = {
    URLS_QUEUE:     write_url,
    RAW_QUEUE:      write_raw_place,
    ANALYSIS_QUEUE: write_analysis,
}


def _worker_loop():
    log.info(f"[worker] DB Writer başladı — kuyruklar: {_QUEUES}")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    while _worker_stats["running"]:
        item = r.brpop(_QUEUES, timeout=5)
        if item is None:
            continue

        queue_name, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.error(f"[worker] Geçersiz JSON ({queue_name}), atlanıyor.")
            continue

        log.info(f"[worker] {queue_name} → işleniyor")
        try:
            _HANDLERS[queue_name](payload)
            _worker_stats["processed"] += 1
        except Exception as e:
            log.error(f"[worker] Hata ({queue_name}): {e}")
            _worker_stats["failed"] += 1

    log.info("[worker] DB Writer durduruldu.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_pool()
    _migrate()
    _worker_stats["running"] = True
    threading.Thread(target=_worker_loop, daemon=True).start()
    log.info("[startup] DB Writer worker otomatik başlatıldı.")
    yield
    _worker_stats["running"] = False
    if _pool:
        _pool.closeall()


app = FastAPI(title="DB Writer", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/worker/status")
def worker_status():
    pending = {}
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        for q in _QUEUES:
            pending[q] = r.llen(q)
    except Exception:
        pass
    return {**_worker_stats, "queues": pending}


@app.post("/worker/stop")
def worker_stop():
    _worker_stats["running"] = False
    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)