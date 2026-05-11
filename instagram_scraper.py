"""
instagram_scraper.py
--------------------
Instagram profilinden son N postun caption + bio bilgisini Playwright ile çeker.
Cookie dosyası ile çalışır — bir kere login ol, cookie'yi kaydet, tekrar kullan.

İlk kurulum (bir kere):
    python instagram_scraper.py --save-cookies

Kullanım:
    python instagram_scraper.py --url https://www.instagram.com/kemun.ankara/
    python instagram_scraper.py --input scraped_data.json --output enriched_data.json

FastAPI servisi:
    uvicorn instagram_scraper:app --port 8084
    POST /scrape  {"url": "https://www.instagram.com/kemun.ankara/"}
    POST /run     {"input": "scraped_data.json", "output": "enriched_data.json"}
    GET  /status
    GET  /health
"""

import json
import logging
import os
import sys
import threading
import time
import argparse
from datetime import datetime, UTC
from pathlib import Path
from urllib.parse import urlparse

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from playwright.sync_api import sync_playwright

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ENRICHER_INPUT   = os.getenv("ENRICHER_INPUT",  "scraped_data.json")
ENRICHER_OUTPUT  = os.getenv("ENRICHER_OUTPUT", "enriched_data.json")
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379")
ENRICHER_QUEUE   = "queue:places:enricher"
ANALYZER_QUEUE   = "queue:places:analyzer"
INDEXER_QUEUE    = "queue:places:indexer"
MAX_POSTS        = int(os.getenv("INSTAGRAM_MAX_POSTS", "10"))
IG_SLEEP         = float(os.getenv("INSTAGRAM_SLEEP", "4.0"))
IG_COOKIES_FILE  = os.getenv("IG_COOKIES_FILE", "instagram_cookies.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_handle(url: str) -> str | None:
    """Her türlü Instagram URL'sinden handle çıkarır."""
    if not url or "instagram.com" not in url:
        return None
    # /profilecard/, /p/, /reel/ gibi path'leri atla
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts:
        return None
    handle = parts[0]
    # Bunlar profil sayfası değil
    if handle in ("p", "reel", "reels", "stories", "explore", "accounts"):
        return None
    return handle


def _launch_context(playwright, headless: bool = True):
    browser = playwright.chromium.launch(
        headless=headless,
        args=["--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled"],
    )
    context_opts = {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "viewport": {"width": 1280, "height": 900},
    }
    context = browser.new_context(**context_opts)
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    if Path(IG_COOKIES_FILE).exists():
        with open(IG_COOKIES_FILE, encoding="utf-8") as f:
            context.add_cookies(json.load(f))
        log.info(f"Cookie yüklendi: {IG_COOKIES_FILE}")
    else:
        log.warning("Cookie dosyası bulunamadı. --save-cookies ile önce login ol.")

    return browser, context


# ---------------------------------------------------------------------------
# Cookie kaydetme (bir kere çalıştırılır)
# ---------------------------------------------------------------------------

def save_cookies():
    """Tarayıcı açar, Instagram'a manuel login yaparsın, cookie kaydedilir."""
    print("\nTarayıcı açılıyor — Instagram'a giriş yap, sonra Enter'a bas...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.instagram.com/accounts/login/")
        input("\nLogin yaptıktan sonra Enter'a bas: ")
        cookies = context.cookies()
        with open(IG_COOKIES_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        browser.close()
    print(f"Cookie kaydedildi → {IG_COOKIES_FILE}")


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def scrape_profile(handle: str) -> dict:
    with sync_playwright() as p:
        browser, context = _launch_context(p, headless=True)
        page = context.new_page()
        page.set_default_timeout(15000)

        url = f"https://www.instagram.com/{handle}/"
        log.info(f"  Açılıyor: {url}")
        page.goto(url)
        page.wait_for_timeout(3000)

        # Bio
        bio = ""
        try:
            bio = page.locator('header section div span, header section div div span').first.inner_text(timeout=5000).strip()
        except Exception:
            pass

        # Takipçi sayısı
        followers = 0
        try:
            for el in page.locator('header section ul li').all():
                text = el.inner_text()
                if "takipçi" in text.lower() or "follower" in text.lower():
                    num_text = el.locator("span, a").first.get_attribute("title") or el.locator("span").nth(1).inner_text()
                    followers = int(num_text.replace(".", "").replace(",", "").strip())
                    break
        except Exception:
            pass

        # Postlar
        posts = []
        try:
            # Sayfa scroll ederek postları yükle
            for _ in range(3):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(1500)

            post_links = page.locator('article a[href*="/p/"]').all()
            for link in post_links[:MAX_POSTS]:
                try:
                    href = link.get_attribute("href")
                    if not href:
                        continue
                    post_url = f"https://www.instagram.com{href}"
                    page.goto(post_url)
                    page.wait_for_timeout(2000)

                    caption = ""
                    try:
                        caption = page.locator('article div[role="presentation"] span, h1').first.inner_text(timeout=4000).strip()
                    except Exception:
                        pass

                    date = ""
                    try:
                        date = page.locator('time').first.get_attribute("datetime", timeout=3000) or ""
                        if date:
                            date = date[:10]  # YYYY-MM-DD
                    except Exception:
                        pass

                    if caption:
                        posts.append({"date": date, "caption": caption})

                    page.go_back()
                    page.wait_for_timeout(1500)
                except Exception as e:
                    log.debug(f"Post hatası: {e}")
                    continue
        except Exception as e:
            log.warning(f"Post çekme hatası: {e}")

        browser.close()

    return {
        "handle": handle,
        "bio": bio,
        "followers": followers,
        "posts": posts,
        "scraped_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_place(place: dict) -> dict:
    if place.get("website_type") != "instagram":
        return place

    handle = extract_handle(place.get("website_url", ""))
    if not handle:
        return place

    try:
        log.info(f"  Instagram: @{handle}")
        ig = scrape_profile(handle)
        place["instagram"] = ig
        log.info(f"  ✓ {len(ig['posts'])} post | {ig['followers']} takipçi")
    except Exception as e:
        log.warning(f"  Hata @{handle}: {e}")

    return place


def run(input_path: str, output_path: str):
    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Dosya bulunamadı: {input_file}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    ig_count = sum(1 for p in data if p.get("website_type") == "instagram")
    log.info(f"Toplam {len(data)} mekan | Instagram olan: {ig_count}")

    for i, place in enumerate(data, 1):
        if place.get("website_type") != "instagram":
            continue
        log.info(f"[{i}/{len(data)}] {place.get('name', '?')}")
        enrich_place(place)
        time.sleep(IG_SLEEP)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Kaydedildi → {output_path}")


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Instagram Enricher")
_job: dict = {"status": "idle", "detail": ""}
_lock = threading.Lock()
_worker_stats = {"processed": 0, "failed": 0, "running": False}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scrape")
def scrape_url(data: dict):
    """Tek bir Instagram URL'sini anında scrape eder."""
    url = data.get("url", "")
    handle = extract_handle(url)
    if not handle:
        return {"error": "Geçerli bir Instagram profil URL'si değil"}
    try:
        result = scrape_profile(handle)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/run")
def trigger(data: dict = None):
    if data is None:
        data = {}
    inp = data.get("input", ENRICHER_INPUT)
    out = data.get("output", ENRICHER_OUTPUT)

    with _lock:
        if _job["status"] == "running":
            return {"status": "already_running"}
        _job.update({"status": "running", "detail": ""})

    def _task():
        try:
            run(inp, out)
            with _lock:
                _job["status"] = "done"
        except Exception as e:
            with _lock:
                _job.update({"status": "error", "detail": str(e)})

    threading.Thread(target=_task, daemon=True).start()
    return {"status": "started", "input": inp, "output": out}


@app.get("/status")
def status():
    with _lock:
        return dict(_job)


# ---------------------------------------------------------------------------
# Redis worker
# ---------------------------------------------------------------------------

def _worker_loop():
    log.info(f"[worker] Enricher Redis worker başladı — kuyruk: {ENRICHER_QUEUE}")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    while _worker_stats["running"]:
        item = r.brpop(ENRICHER_QUEUE, timeout=5)
        if item is None:
            continue

        _, raw = item
        try:
            place = json.loads(raw)
        except json.JSONDecodeError:
            continue

        log.info(f"[worker] İşleniyor: '{place.get('name', '?')}'")
        enriched = enrich_place(place)
        _worker_stats["processed"] += 1

        payload = json.dumps(enriched, ensure_ascii=False)
        r.lpush(ANALYZER_QUEUE, payload)
        r.lpush(INDEXER_QUEUE, payload)
        time.sleep(IG_SLEEP)

    log.info("[worker] Enricher worker durduruldu.")


@app.on_event("startup")
def auto_start_worker():
    _worker_stats["running"] = True
    threading.Thread(target=_worker_loop, daemon=True).start()
    log.info("[startup] Enricher Redis worker otomatik başlatıldı.")


@app.post("/worker/stop")
def worker_stop():
    _worker_stats["running"] = False
    return {"status": "stopping"}


@app.get("/worker/status")
def worker_status():
    pending = 0
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        pending = r.llen(ENRICHER_QUEUE)
    except Exception:
        pass
    return {**_worker_stats, "queue_pending": pending}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instagram enricher")
    parser.add_argument("--input",        default=None)
    parser.add_argument("--output",       default=ENRICHER_OUTPUT)
    parser.add_argument("--url",          default=None, help="Tek profil scrape et")
    parser.add_argument("--save-cookies", action="store_true", help="Instagram cookie kaydet")
    args = parser.parse_args()

    if args.save_cookies:
        save_cookies()
    elif args.url:
        handle = extract_handle(args.url)
        if not handle:
            print("Geçerli bir Instagram profil URL'si değil")
            sys.exit(1)
        result = scrape_profile(handle)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run(args.input or ENRICHER_INPUT, args.output)