import json
import os

import redis
import uvicorn
from fastapi import FastAPI
from playwright.sync_api import Page, sync_playwright

from URLs import URL

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
ANALYZER_QUEUE = "queue:places:analyzer"
INDEXER_QUEUE  = "queue:places:indexer"

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


def startGoogleMapsScrapper(latitude: str, longitude: str, keyword: str, placeLimit: int):
    with sync_playwright() as playwright:
        page = launchPage(playwright, latitude, longitude, keyword)

        goto(page, URL.GOOGLE_MAPS_BASE_URL.format_url(latitude=latitude, longitude=longitude, keyword=keyword))

        # page.pause()

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

    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
    )
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    page = context.new_page()

    return page


def goto(page: Page, domainName: str):
    page.goto(domainName)


def itarateOverAllPlacesFound(page: Page, placelimit: int):
    feed_selector = 'div[role="feed"]'
    target_urls = []
    all_extracted_data = []

    while len(target_urls) < placelimit:
        places = page.locator(f'{feed_selector} a[href*="/maps/place/"]').all()
        for place in places:
            url = place.get_attribute("href")
            if url and url not in target_urls:
                target_urls.append(url)
                if len(target_urls) >= placelimit: break
        if len(target_urls) >= placelimit: break
        page.locator(feed_selector).evaluate("node => node.scrollBy(0, 2500)")
        page.wait_for_timeout(2000)

    for idx, url in enumerate(target_urls, 1):
        try:
            print(f"\n[{idx}/{len(target_urls)}] Mekana gidiliyor...")
            page.goto(url)
            page.wait_for_timeout(3000)

            place_name = ""
            try:
                place_name = page.locator('h1').first.inner_text(timeout=3000)
                print(f"  Mekan: {place_name}")
            except Exception:
                pass

            reviews = get_reviews_for_place(page, max_reviews=100)

            place_data = {"url": url, "name": place_name, "total_reviews_scraped": len(reviews), "reviews": reviews}

            all_extracted_data.append(place_data)

            try:
                r = get_redis()
                payload = json.dumps(place_data, ensure_ascii=False)
                r.lpush(ANALYZER_QUEUE, payload)
                r.lpush(INDEXER_QUEUE, payload)
                print(f"  Kuyruğa eklendi: {place_name}")
            except Exception as e:
                print(f"  Redis hatası (devam ediliyor): {e}")

        except Exception as e:
            print(f"Hata ({url}): {e}")

    save_data_to_json(all_extracted_data, filename=SCRAPER_OUTPUT)
    return all_extracted_data


def save_data_to_json(data, filename="scraped_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n--- Datas are saved into the file {filename}  ---")


def get_reviews_for_place(page: Page, max_reviews: int = 200):
    try:
        reviews_tab = page.locator('button[role="tab"]:has-text("Yorumlar"), button[role="tab"]:has-text("Reviews")')

        if reviews_tab.count() > 0:
            reviews_tab.first.click()
            page.wait_for_timeout(2000)
            print("  Yorumlar sekmesine geçildi.")
        else:
            print("  Yorumlar sekmesi bulunamadı, sayfa üzerindeki yorumlar denenecek.")

        # Yorumların bulunduğu kaydırılabilir panel
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
                page.wait_for_timeout(800)  # buton tıklaması sonrası DOM güncellemesini bekle

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

            # Panel içinde scroll — sayfa scroll'u değil
            try:
                review_panel.evaluate("node => node.scrollBy(0, 3000)")
            except Exception:
                page.mouse.wheel(0, 3000)
            page.wait_for_timeout(2500)  # scroll sonrası yeni içeriklerin yüklenmesini bekle

        print(f"  Tamamlandi: {len(collected_reviews[:max_reviews])} yorum çekildi.")
        return collected_reviews[:max_reviews]

    except Exception as e:
        print(f"Yorum çekilirken hata oluştu: {str(e)}")
        return []


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
