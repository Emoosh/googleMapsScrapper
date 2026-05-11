import json
import os
import re
import sys

from playwright.sync_api import sync_playwright

INPUT_FILE  = os.getenv("INPUT_FILE",  "pending_urls.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "scraped_data.json")
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "100"))


def load_urls() -> list[str]:
    with open(INPUT_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_existing(filename: str) -> list[dict]:
    if os.path.exists(filename):
        try:
            with open(filename, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save(data: list[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def launch_page(playwright):
    browser = playwright.chromium.launch(
        headless=os.getenv("HEADLESS", "false").lower() == "true",
        args=["--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled", "--window-size=1920,1080"],
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


def get_reviews(page, max_reviews: int) -> list[str]:
    try:
        reviews_tab = page.locator('button[role="tab"]:has-text("Yorumlar"), button[role="tab"]:has-text("Reviews")')
        if reviews_tab.count() > 0:
            reviews_tab.first.click()
            page.wait_for_timeout(2000)

        review_panel = page.locator('div.m6QErb.DxyBCb').first
        collected = []
        last_count = 0
        scroll_attempts = 0

        while len(collected) < max_reviews:
            for btn in page.locator('button:has-text("Tamamını oku"), button:has-text("More")').all():
                try:
                    if btn.is_visible():
                        btn.click(timeout=500)
                except Exception:
                    pass

            for review in page.locator('span.wiI7pd').all_text_contents():
                if review not in collected:
                    collected.append(review)
                    if len(collected) >= max_reviews:
                        break

            if len(collected) == last_count:
                scroll_attempts += 1
                if scroll_attempts > 8:
                    break
            else:
                scroll_attempts = 0
            last_count = len(collected)

            try:
                review_panel.evaluate("node => node.scrollBy(0, 3000)")
            except Exception:
                page.mouse.wheel(0, 3000)
            page.wait_for_timeout(2500)

        return collected[:max_reviews]
    except Exception:
        return []


def detect_link_type(url: str) -> str:
    if not url:
        return "unknown"
    if "instagram.com" in url:
        return "instagram"
    if "facebook.com" in url:
        return "facebook"
    if "tripadvisor.com" in url:
        return "tripadvisor"
    return "website"


def extract_external_links(page) -> dict:
    result = {"website_url": None, "website_type": None}
    try:
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
                    result["website_type"] = detect_link_type(href)
                    break
    except Exception:
        pass
    return result


def get_images(page, max_images: int = 10) -> list[str]:
    try:
        photos_btn = page.locator('button[aria-label*="Fotoğraf"], button[aria-label*="Photo"]').first
        if photos_btn.count() > 0:
            photos_btn.click()
            page.wait_for_timeout(2000)

        urls = []
        for el in page.locator('button[style*="background-image"]').all():
            style = el.get_attribute("style") or ""
            match = re.search(r'url\("?(https?://[^")\s]+)"?\)', style)
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

        return urls[:max_images]
    except Exception:
        return []


def main():
    urls = load_urls()
    already_done = load_existing(OUTPUT_FILE)
    done_urls = {p["url"] for p in already_done}

    remaining = [u for u in urls if u not in done_urls]
    total = len(urls)
    skipped = len(urls) - len(remaining)

    print(f"Toplam URL: {total} | Zaten scraped: {skipped} | Kalan: {len(remaining)}")
    if not remaining:
        print("Tümü tamamlanmış.")
        return

    results = list(already_done)

    with sync_playwright() as playwright:
        page = launch_page(playwright)

        for idx, url in enumerate(remaining, 1):
            print(f"\n[{idx}/{len(remaining)}] scrape ediliyor...")
            try:
                page.goto(url)
                page.wait_for_timeout(3000)

                current_url = page.url
                if "/contrib/" in current_url or "/maps/place/" not in current_url:
                    print(f"  Yönlendirme tespit edildi, atlanıyor: {current_url}")
                    continue

                name = ""
                try:
                    name = page.locator('h1').first.inner_text(timeout=3000)
                except Exception:
                    pass

                reviews = get_reviews(page, MAX_REVIEWS)
                images = get_images(page, max_images=10)
                links = extract_external_links(page)
                print(f"  {name} | {len(reviews)} yorum | {len(images)} resim | site: {links['website_type']}")

                results.append({
                    "url": url,
                    "name": name,
                    "website_url": links["website_url"],
                    "website_type": links["website_type"],
                    "total_reviews_scraped": len(reviews),
                    "reviews": reviews,
                    "images": images,
                })
                save(results, OUTPUT_FILE)
            except Exception as e:
                print(f"  Hata: {e}")

    print(f"\nTamamlandı. {len(results)} mekan → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()