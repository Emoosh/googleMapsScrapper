"""
comment_analyzer.py
-------------------
Mikroservis: scraped_data.json dosyasını okur, her mekan için
Gemini API ile yorum analizi yapar ve analyzed_data.json olarak kaydeder.

Kullanım:
    python comment_analyzer.py
    python comment_analyzer.py --input baska_dosya.json --output sonuc.json
    python comment_analyzer.py --dry-run   # API çağrısı yapmadan yapıyı test et
"""

import json
import os
import re
import time
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, UTC
from dotenv import load_dotenv
from urllib.parse import unquote

from google import genai

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("comment_analyzer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-3-flash-preview"
RATE_LIMIT_DELAY = 1.5   # saniye — mekanlar arası bekleme
MAX_RETRIES      = 3
RETRY_DELAY      = 5     # saniye — hata sonrası bekleme

TARGET_SCHEMA = {
    "place_name": "string",
    "source_url": "string",
    "total_reviews_scraped": "integer",
    "analyzed_at": "ISO datetime string",
    "scores": {
        "atmosfer": "float 1-10",
        "kahve_veya_yemek_kalitesi": "float 1-10",
        "tatlilar": "float 1-10",
        "hizmet": "float 1-10",
        "sessizlik_calisma_uygunlugu": "float 1-10",
        "fiyat_performans": "float 1-10",
    },
    "genel_puan": "float 1-10",
    "ozet": "string — 2-3 cümle nesnel özet",
    "one_cikanlar": ["string — en fazla 5 madde"],
    "eksiler": ["string — yorumlarda geçen somut şikayetler"],
    "populer_urunler": ["string — yorumlarda adı geçen ürünler"],
    "etiketler": ["string — kısa tanımlayıcı etiketler"],
    "kim_icin_ideal": "string — tek cümle",
    "fiyat_seviyesi": "ucuz | orta | orta-üst | pahalı | belirtilmemiş",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_gemini_response(raw: str) -> str:
    """Gemini bazen ```json ``` bloğu döndürür — temizle."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return raw


def build_prompt(place_name: str, reviews: list[str]) -> str:
    reviews_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    schema_str   = json.dumps(TARGET_SCHEMA, ensure_ascii=False, indent=2)

    return f"""
Aşağıdaki mekan yorumlarını analiz et ve YALNIZCA geçerli bir JSON objesi döndür.
Başka hiçbir şey yazma. Markdown, açıklama veya kod bloğu kullanma.

Mekan adı: {place_name}

Yorumlar:
{reviews_text}

Döndürmen gereken JSON formatı:
{schema_str}

Kurallar:
- Tüm puanlar 1-10 arasında float (örn: 8.5)
- ozet: 2-3 cümle, nesnel, yalnızca yorumlara dayalı
- one_cikanlar: en fazla 5 madde, yorumlarda geçen güçlü yönler
- eksiler: yorumlarda geçen somut şikayetler; yoksa boş liste []
- populer_urunler: yorumlarda adı geçen yiyecek/içecekler
- etiketler: mekanı tanımlayan kısa kelimeler (örn: "cozy", "bahçeli", "çalışma dostu")
- fiyat_seviyesi: yorumlardaki ipuçlarına göre kategorize et
- analyzed_at ve source_url alanlarını boş bırak, kod dolduracak
"""


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

def analyze_single(
    model,
    place_name: str,
    reviews: list[str],
    dry_run: bool = False,
) -> dict:
    """Tek bir mekanı analiz eder, ham dict döndürür."""

    if dry_run:
        log.info(f"  [dry-run] '{place_name}' için API çağrısı atlanıyor.")
        return {
            "place_name": place_name,
            "scores": {k: 0.0 for k in TARGET_SCHEMA["scores"]},
            "genel_puan": 0.0,
            "ozet": "dry-run modu",
            "one_cikanlar": [],
            "eksiler": [],
            "populer_urunler": [],
            "etiketler": ["dry-run"],
            "kim_icin_ideal": "dry-run",
            "fiyat_seviyesi": "belirtilmemiş",
        }

    prompt = build_prompt(place_name, reviews)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw      = clean_gemini_response(response.text)
            parsed   = json.loads(raw)
            return parsed
        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse hatası (deneme {attempt}/{MAX_RETRIES}): {e}")
            wait = RETRY_DELAY
        except Exception as e:
            err_str = str(e)
            log.warning(f"  API hatası (deneme {attempt}/{MAX_RETRIES}): {e}")

            # 429 ise API'nin önerdiği retryDelay'i kullan
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                retry_match = re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(retry_match.group(1)) + 2 if retry_match else 60
                log.info(f"  Kota aşıldı — {wait}s bekleniyor (API önerisi)...")
            else:
                wait = RETRY_DELAY

        if attempt < MAX_RETRIES:
            log.info(f"  {wait}s beklenip tekrar deneniyor...")
            time.sleep(wait)

    raise RuntimeError(f"'{place_name}' için {MAX_RETRIES} denemede de analiz başarısız.")

def extract_place_name(entry: dict) -> str:
    if entry.get("place_name"):
        return entry["place_name"]
    url = entry.get("url", "")
    match = re.search(r"/place/([^/]+)", url)
    if match:
        raw_name = match.group(1).replace("+", " ")
        return unquote(raw_name)  # %C3%BC → ü
    return "Bilinmeyen Mekan"
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, output_path: str, dry_run: bool = False):
    # --- Girdi dosyasını oku ---
    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        scraped_data = json.load(f)

    if not isinstance(scraped_data, list):
        log.error("scraped_data.json bir liste (array) olmalı.")
        sys.exit(1)

    total = len(scraped_data)
    log.info(f"{total} mekan bulundu → analiz başlıyor.")

    # --- Gemini istemcisini başlat ---
    if not dry_run:
        if not GEMINI_API_KEY:
            log.error(
                "GEMINI_API_KEY bulunamadı!\n"
                ".env dosyanızın proje klasöründe olduğundan ve şu satırı içerdiğinden emin olun:\n"
                "  GEMINI_API_KEY=AIzaSy..."
            )
            sys.exit(1)
        model = genai.Client(api_key=GEMINI_API_KEY)
        log.info(f"Gemini client başlatıldı. Model: {GEMINI_MODEL}")
    else:
        model = None
    # --- Her mekanı işle ---
    results      = []
    failed       = []

    for idx, entry in enumerate(scraped_data, start=1):
        place_name = extract_place_name(entry)
        reviews    = entry.get("reviews", [])

        log.info(f"[{idx}/{total}] '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning(f"  Yorum bulunamadı, atlanıyor.")
            failed.append({"place_name": place_name, "reason": "yorum yok"})
            continue

        try:
            result = analyze_single(model, place_name, reviews, dry_run=dry_run)

            # Kod tarafından doldurulan alanlar
            result["place_name"]            = place_name
            result["source_url"]            = entry.get("url", "")
            result["total_reviews_scraped"] = entry.get("total_reviews_scraped", len(reviews))
            result["analyzed_at"]           = datetime.now(UTC).isoformat()

            results.append(result)
            log.info(f"  ✓ Genel puan: {result.get('genel_puan', '?')}")

        except RuntimeError as e:
            log.error(f"  ✗ {e}")
            failed.append({"place_name": place_name, "reason": str(e)})

        # Rate limit — son mekan değilse bekle
        if idx < total and not dry_run:
            time.sleep(RATE_LIMIT_DELAY)

    # --- Çıktıyı kaydet ---
    output_file = Path(output_path)
    output_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_places": total,
        "successful": len(results),
        "failed": len(failed),
        "failed_places": failed,
        "data": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    log.info(
        f"\nTamamlandı → {output_file} | "
        f"Başarılı: {len(results)}/{total} | "
        f"Başarısız: {len(failed)}/{total}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafe yorum analiz mikroservisi")
    parser.add_argument(
        "--input",  default="scraped_data.json",
        help="Girdi dosyası (varsayılan: scraped_data.json)"
    )
    parser.add_argument(
        "--output", default="analyzed_data.json",
        help="Çıktı dosyası (varsayılan: analyzed_data.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="API çağrısı yapmadan pipeline'ı test et"
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
    )