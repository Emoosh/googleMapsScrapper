import json

import uvicorn
from fastapi import FastAPI
from playwright.sync_api import Page, sync_playwright

from URLs import URL

app = FastAPI()


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
    browser = playwright.chromium.launch(headless=False)

    page = browser.new_page()

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

    for url in target_urls:
        try:
            print(f"İşleniyor: {url}")
            page.goto(url)
            page.wait_for_timeout(3000)

            reviews = get_reviews_for_place(page, max_reviews=10)

            place_data = {"url": url, "total_reviews_scraped": len(reviews), "reviews": reviews}

            all_extracted_data.append(place_data)

        except Exception as e:
            print(f"Hata ({url}): {e}")

    save_data_to_json(all_extracted_data)
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
        else:
            print("!!")

        collected_reviews = []
        last_count = 0
        scroll_attempts = 0

        print(f"Yorumlar toplanıyor (Hedef: {max_reviews})...")

        while len(collected_reviews) < max_reviews:
            more_buttons = page.locator('button:has-text("Tamamını oku"), button:has-text("More")').all()
            for btn in more_buttons:
                try:
                    if btn.is_visible():
                        btn.click(timeout=500)
                except:
                    pass

            elements = page.locator('span.wiI7pd').all_text_contents()

            for review in elements:
                if review not in collected_reviews:
                    collected_reviews.append(review)
                    if len(collected_reviews) >= max_reviews:
                        break

            if len(collected_reviews) == last_count:
                scroll_attempts += 1
                if scroll_attempts > 5:
                    print("Daha fazla yeni yorum bulunamadı. Kaydırma durduruluyor.")
                    break
            else:
                scroll_attempts = 0

            last_count = len(collected_reviews)

            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(2000)  # Verilerin yüklenmesi için zaman tanı

        print(f"İşlem Tamamlandı: {len(collected_reviews[:max_reviews])} adet yorum çekildi.")
        return collected_reviews[:max_reviews]

    except Exception as e:
        print(f"Yorum çekilirken hata oluştu: {str(e)}")
        return []


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
