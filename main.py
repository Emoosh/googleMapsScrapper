import json
import math
import os

import redis
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from playwright.sync_api import Page, sync_playwright

from URLs import URL

REDIS_URL           = os.getenv("REDIS_URL", "redis://localhost:6379")
ANALYZER_QUEUE      = "queue:places:analyzer"
INDEXER_QUEUE       = "queue:places:indexer"
SCRAPED_URLS_KEY    = "scraped:urls"
SCRAPED_CELLS_KEY   = "scraped:cells"
PENDING_URLS_KEY    = "pending:urls"
SCRAPE_PHASE_KEY    = "scrape:phase"

ANKARA_BOUNDS = {
    "lat_min": 39.75, "lat_max": 40.05,
    "lon_min": 32.50, "lon_max": 33.10,
}

def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="Scraper Mikroservisi")

SCRAPER_OUTPUT = os.getenv("SCRAPER_OUTPUT", "scraped_data.json")


@app.post("/api/googleMaps/service")
def getDataFromTheServer(data: dict):
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    keyword = data.get("keyword")
    placeLimit = data.get("placeLimit")
    print("New Search Terms Triggered -> " + latitude + "," + longitude + "," + keyword + "," + placeLimit + "\n")
    startGoogleMapsScrapper(latitude, longitude, keyword, int(placeLimit))


@app.post("/api/googleMaps/city/cafes")
def scrape_city_cafes(
    background_tasks: BackgroundTasks,
    data: dict = None,
):
    if data is None:
        data = {}
    keyword     = data.get("keyword", "kafe")
    divisions   = int(data.get("divisions", 4))
    max_reviews = int(data.get("max_reviews", 20))
    resume      = bool(data.get("resume", True))
    bounds      = data.get("bounds", ANKARA_BOUNDS)

    grid = _build_grid_divisions(bounds, divisions)
    background_tasks.add_task(
        _scrape_city_grid, grid, keyword, resume, max_reviews
    )
    return {
        "status": "started",
        "grid_cells": len(grid),
        "divisions": f"{divisions}x{divisions}",
        "resume": resume,
    }


@app.get("/api/googleMaps/city/status")
def scrape_city_status():
    r = get_redis()
    phase         = r.get(SCRAPE_PHASE_KEY) or "idle"
    pending       = r.llen(PENDING_URLS_KEY)
    scraped       = r.scard(SCRAPED_URLS_KEY)
    cells_done    = r.scard(SCRAPED_CELLS_KEY)
    place_count   = 0
    if os.path.exists(SCRAPER_OUTPUT):
        try:
            with open(SCRAPER_OUTPUT, encoding="utf-8") as f:
                place_count = len(json.load(f))
        except Exception:
            pass
    return {
        "phase": phase,
        "cells_completed": cells_done,
        "urls_found": scraped + pending,
        "urls_scraped": scraped,
        "urls_pending": pending,
        "places_in_file": place_count,
    }


def startGoogleMapsScrapper(latitude: str, longitude: str, keyword: str, placeLimit: int):
    with sync_playwright() as playwright:
        page = launchPage(playwright, latitude, longitude, keyword)
        goto(page, URL.GOOGLE_MAPS_BASE_URL.format_url(latitude=latitude, longitude=longitude, keyword=keyword))
        itarateOverAllPlacesFound(page, placeLimit)


def launchPage(playwright, longitude: str, latitude: str, keyword: str):
    headless = os.getenv("HEADLESS", "false").lower() == "true"
    browser = playwright.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--window-size=1920,1080",
        ],
    )
    proxy_url = os.getenv("PROXY_URL", "")
    context_opts = {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "viewport": {"width": 1920, "height": 1080},
    }
    if proxy_url:
        context_opts["proxy"] = {"server": proxy_url}
        print(f"  Proxy aktif: {proxy_url}")
    context = browser.new_context(**context_opts)
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    cookies_file = os.getenv("COOKIES_FILE", "cookies.json")
    if os.path.exists(cookies_file):
        with open(cookies_file, encoding="utf-8") as f:
            cookies = json.load(f)
        context.add_cookies(cookies)
        print(f"  Cookie yüklendi: {cookies_file} ({len(cookies)} adet)")

    page = context.new_page()
    page.set_default_navigation_timeout(30000)
    page.set_default_timeout(15000)
    return page


def goto(page: Page, domainName: str):
    page.goto(domainName)


# ─── Faz 1: Feed'den URL topla ───────────────────────────────────────────────

def _collect_urls_from_feed(page: Page, seen_urls: set) -> list[str]:
    feed_selector = 'div[role="feed"]'
    collected = []
    last_total = 0
    no_new_attempts = 0

    while True:
        places = page.locator(f'{feed_selector} a[href*="/maps/place/"]').all()
        current_total = len(places)
        for place in places:
            url = place.get_attribute("href")
            if not url or url in seen_urls or url in collected:
                continue
            collected.append(url)

        end_of_list = page.locator(
            'span:has-text("Bu listenin sonuna geldiniz"), span:has-text("You\'ve reached the end of the list")'
        )
        if end_of_list.count() > 0:
            print(f"  Liste sonu — {len(collected)} yeni URL")
            break

        if current_total == last_total:
            no_new_attempts += 1
            if no_new_attempts >= 4:
                print(f"  Feed durdu — {len(collected)} yeni URL")
                break
        else:
            no_new_attempts = 0

        last_total = current_total
        try:
            page.locator(feed_selector).evaluate("node => node.scrollBy(0, 2500)")
        except Exception:
            no_new_attempts += 1
        page.wait_for_timeout(2000)

    return collected


# ─── Faz 2: Mekanı scrape et ─────────────────────────────────────────────────

def _detect_link_type(url: str) -> str:
    if not url:
        return "unknown"
    if "instagram.com" in url:
        return "instagram"
    if "facebook.com" in url:
        return "facebook"
    if "tripadvisor.com" in url:
        return "tripadvisor"
    return "website"


def _get_images(page: Page, max_images: int = 10) -> list[str]:
    urls = []
    try:
        photos_btn = page.locator('button[aria-label*="Fotoğraf"], button[aria-label*="Photo"]').first
        if photos_btn.count() > 0:
            photos_btn.click()
            page.wait_for_timeout(2000)

        for el in page.locator('button[style*="background-image"]').all():
            style = el.get_attribute("style") or ""
            import re as _re
            match = _re.search(r'url\("?(https?://[^")\s]+)"?\)', style)
            if match:
                img_url = match.group(1)
                if img_url not in urls:
                    urls.append(img_url)
            if len(urls) >= max_images:
                break

        if not urls:
            for img in page.locator('img[src*="googleusercontent"]').all():
                src = img.get_attribute("src") or ""
                if src and src not in urls:
                    urls.append(src)
                if len(urls) >= max_images:
                    break
    except Exception:
        pass
    return urls[:max_images]


def _extract_external_links(page: Page) -> dict:
    result = {"website_url": None, "website_type": None}
    try:
        # Google Maps website butonu: data-item-id="authority" veya aria-label içeriği
        selectors = [
            'a[data-item-id="authority"]',
            'a[aria-label*="web" i]',
            'a[aria-label*="site" i]',
        ]
        for sel in selectors:
            el = page.locator(sel).first
            if el.count() > 0:
                href = el.get_attribute("href")
                if href and href.startswith("http"):
                    result["website_url"] = href
                    result["website_type"] = _detect_link_type(href)
                    break
    except Exception:
        pass
    return result


def _scrape_place(page: Page, url: str, max_reviews: int) -> dict:
    page.goto(url)
    page.wait_for_timeout(3000)

    place_name = ""
    try:
        place_name = page.locator('h1').first.inner_text(timeout=3000)
    except Exception:
        pass

    links = _extract_external_links(page)
    reviews = get_reviews_for_place(page, max_reviews=max_reviews)
    images = _get_images(page, max_images=10)
    return {
        "url": url,
        "name": place_name,
        "website_url": links["website_url"],
        "website_type": links["website_type"],
        "images": images,
        "total_reviews_scraped": len(reviews),
        "reviews": reviews,
    }


# ─── Grid tarama ─────────────────────────────────────────────────────────────

def _scrape_city_grid(
    grid: list,
    keyword: str,
    resume: bool,
    max_reviews: int = 20,
):
    r = get_redis()

    # ── FAZ 1: URL toplama ──
    r.set(SCRAPE_PHASE_KEY, "faz-1: URL toplaniyor")
    total_cells = len(grid)

    if not resume:
        r.delete(PENDING_URLS_KEY)
        r.delete(SCRAPED_CELLS_KEY)
        r.delete(SCRAPED_URLS_KEY)

    already_seen: set = set(r.smembers(SCRAPED_URLS_KEY)) | set(r.lrange(PENDING_URLS_KEY, 0, -1))

    for idx, cell in enumerate(grid, 1):
        lat, lon, cell_key = cell["lat"], cell["lon"], cell["key"]
        if r.sismember(SCRAPED_CELLS_KEY, cell_key):
            print(f"[Faz 1] Hücre {idx}/{total_cells} atlandı → ({lat}, {lon})")
            continue

        print(f"\n[Faz 1] Hücre {idx}/{total_cells} → ({lat}, {lon})")
        try:
            with sync_playwright() as playwright:
                page = launchPage(playwright, str(lat), str(lon), keyword)
                goto(page, URL.GOOGLE_MAPS_BASE_URL.format_url(
                    latitude=lat, longitude=lon, keyword=keyword
                ))
                new_urls = _collect_urls_from_feed(page, already_seen)

            if new_urls:
                r.rpush(PENDING_URLS_KEY, *new_urls)
                already_seen.update(new_urls)
                print(f"[Faz 1] +{len(new_urls)} URL | Toplam bekleyen: {r.llen(PENDING_URLS_KEY)}")

            r.sadd(SCRAPED_CELLS_KEY, cell_key)
        except Exception as e:
            print(f"[Faz 1] Hücre hatası ({lat},{lon}): {e}")

    total_found = r.llen(PENDING_URLS_KEY)
    print(f"\n{'='*50}")
    print(f"[Faz 1 Tamamlandı] Toplam {total_found} benzersiz mekan bulundu")
    print(f"{'='*50}\n")

    # ── FAZ 2: Mekan scraping ──
    r.set(SCRAPE_PHASE_KEY, "faz-2: mekanlar scrape ediliyor")

    all_data: list[dict] = []
    if resume and os.path.exists(SCRAPER_OUTPUT):
        try:
            with open(SCRAPER_OUTPUT, encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"[Faz 2] Resume: {len(all_data)} mevcut mekan yüklendi.")
        except Exception:
            pass

    scraped_so_far = len(all_data)
    total_to_scrape = total_found + scraped_so_far

    try:
        with sync_playwright() as playwright:
            page = launchPage(playwright, "0", "0", keyword)
            while True:
                url = r.lpop(PENDING_URLS_KEY)
                if not url:
                    break
                scraped_so_far += 1
                pending_left = r.llen(PENDING_URLS_KEY)
                print(f"\n[Faz 2] {scraped_so_far}/{total_to_scrape} (bekleyen: {pending_left}) → scrape ediliyor")
                try:
                    place_data = _scrape_place(page, url, max_reviews)
                    print(f"  Mekan: {place_data['name']} | {place_data['total_reviews_scraped']} yorum")
                    all_data.append(place_data)
                    save_data_to_json(all_data, SCRAPER_OUTPUT)

                    redis_client = get_redis()
                    payload = json.dumps(place_data, ensure_ascii=False)
                    redis_client.lpush(ANALYZER_QUEUE, payload)
                    redis_client.lpush(INDEXER_QUEUE, payload)
                    redis_client.sadd(SCRAPED_URLS_KEY, url)
                except Exception as e:
                    print(f"  Hata, URL tekrar kuyruğa alındı: {e}")
                    r.rpush(PENDING_URLS_KEY, url)
    except Exception as e:
        print(f"[Faz 2] Kritik hata: {e}")

    r.set(SCRAPE_PHASE_KEY, "tamamlandi")
    print(f"\n[Tamamlandı] {len(all_data)} mekan kaydedildi.")


def save_data_to_json(data, filename="scraped_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"  Dosyaya kaydedildi: {len(data)} mekan")


def _build_grid(bounds: dict, step_km: float) -> list[tuple[float, float]]:
    lat_step = step_km / 111.0
    lon_step = step_km / (111.0 * abs(math.cos(math.radians(
        (bounds["lat_min"] + bounds["lat_max"]) / 2
    ))))
    points = []
    lat = bounds["lat_min"]
    while lat <= bounds["lat_max"]:
        lon = bounds["lon_min"]
        while lon <= bounds["lon_max"]:
            points.append((round(lat, 6), round(lon, 6)))
            lon += lon_step
        lat += lat_step
    return points


def _build_grid_divisions(bounds: dict, divisions: int) -> list[dict]:
    lat_size = (bounds["lat_max"] - bounds["lat_min"]) / divisions
    lon_size = (bounds["lon_max"] - bounds["lon_min"]) / divisions
    cells = []
    for row in range(divisions):
        for col in range(divisions):
            lat_min = bounds["lat_min"] + row * lat_size
            lat_max = lat_min + lat_size
            lon_min = bounds["lon_min"] + col * lon_size
            lon_max = lon_min + lon_size
            cells.append({
                "lat": round((lat_min + lat_max) / 2, 6),
                "lon": round((lon_min + lon_max) / 2, 6),
                "key": f"{row},{col}",
            })
    return cells


def itarateOverAllPlacesFound(page: Page, placelimit: int, scraped_urls_set_key: str = None, max_reviews: int = 20):
    seen: set = set()
    if scraped_urls_set_key:
        r = get_redis()
        seen = set(r.smembers(scraped_urls_set_key))

    target_urls = _collect_urls_from_feed(page, seen)
    if placelimit < 999999:
        target_urls = target_urls[:placelimit]

    all_extracted_data = []
    for idx, url in enumerate(target_urls, 1):
        try:
            print(f"\n[{idx}/{len(target_urls)}] Mekana gidiliyor...")
            place_data = _scrape_place(page, url, max_reviews)
            print(f"  Mekan: {place_data['name']}")
            all_extracted_data.append(place_data)
            try:
                redis_client = get_redis()
                payload = json.dumps(place_data, ensure_ascii=False)
                redis_client.lpush(ANALYZER_QUEUE, payload)
                redis_client.lpush(INDEXER_QUEUE, payload)
                if scraped_urls_set_key:
                    redis_client.sadd(scraped_urls_set_key, url)
                print(f"  Kuyruğa eklendi: {place_data['name']}")
            except Exception as e:
                print(f"  Redis hatası (devam ediliyor): {e}")
        except Exception as e:
            print(f"Hata ({url}): {e}")

    if scraped_urls_set_key is None:
        save_data_to_json(all_extracted_data, filename=SCRAPER_OUTPUT)
    return all_extracted_data


def get_reviews_for_place(page: Page, max_reviews: int = 200):
    try:
        reviews_tab = page.locator('button[role="tab"]:has-text("Yorumlar"), button[role="tab"]:has-text("Reviews")')
        if reviews_tab.count() > 0:
            reviews_tab.first.click()
            page.wait_for_timeout(2000)
            print("  Yorumlar sekmesine geçildi.")
        else:
            print("  Yorumlar sekmesi bulunamadı.")

        review_panel = page.locator('div.m6QErb.DxyBCb').first
        collected_reviews = []
        last_count = 0
        scroll_attempts = 0

        print(f"  Yorumlar toplanıyor (Hedef: {max_reviews})...")

        while len(collected_reviews) < max_reviews:
            more_buttons = page.locator('button:has-text("Tamamını oku"), button:has-text("More")').all()
            clicked = 0
            for btn in more_buttons:
                try:
                    if btn.is_visible():
                        btn.click(timeout=500)
                        clicked += 1
                except Exception:
                    pass
            if clicked > 0:
                page.wait_for_timeout(800)

            elements = page.locator('span.wiI7pd').all_text_contents()
            for review in elements:
                if review not in collected_reviews:
                    collected_reviews.append(review)
                    if len(collected_reviews) >= max_reviews:
                        break

            print(f"  [{len(collected_reviews)}/{max_reviews}] yorum toplandı...")

            if len(collected_reviews) == last_count:
                scroll_attempts += 1
                if scroll_attempts > 8:
                    print("  Daha fazla yeni yorum bulunamadı, durduruluyor.")
                    break
            else:
                scroll_attempts = 0

            last_count = len(collected_reviews)
            try:
                review_panel.evaluate("node => node.scrollBy(0, 3000)")
            except Exception:
                page.mouse.wheel(0, 3000)
            page.wait_for_timeout(2500)

        print(f"  Tamamlandi: {len(collected_reviews[:max_reviews])} yorum çekildi.")
        return collected_reviews[:max_reviews]

    except Exception as e:
        print(f"Yorum çekilirken hata oluştu: {str(e)}")
        return []


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)