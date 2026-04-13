"""
comment_analyzer.py
-------------------
Mikroservis: scraped_data.json dosyasını okur, her mekan için
local Ollama (qwen2.5:14b) ile yorum analizi yapar ve analyzed_data.json olarak kaydeder.

Kullanım:
    python comment_analyzer.py
    python comment_analyzer.py --input baska_dosya.json --output sonuc.json
    python comment_analyzer.py --dry-run   # API çağrısı yapmadan yapıyı test et
    python comment_analyzer.py --model qwen2.5:32b  # farklı model
"""

import json
import os
import re
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, UTC
from urllib.parse import unquote

from openai import OpenAI

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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
MAX_RETRIES     = 3

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

SYSTEM_PROMPT = (
    "Sen bir mekan analiz asistanısın. "
    "Sana verilen Google Maps yorumlarını analiz edip YALNIZCA geçerli bir JSON objesi döndür. "
    "Başka hiçbir şey yazma. Markdown, açıklama veya kod bloğu kullanma."
)


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


def clean_response(raw: str) -> str:
    """Model bazen ```json ``` bloğu döndürür — temizle."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return raw


def build_user_message(place_name: str, reviews: list[str]) -> str:
    reviews_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    schema_str   = json.dumps(TARGET_SCHEMA, ensure_ascii=False, indent=2)
    return f"""Mekan adı: {place_name}

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
- analyzed_at ve source_url alanlarını boş bırak, kod dolduracak"""


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

def analyze_single(
    client: OpenAI,
    model: str,
    place_name: str,
    reviews: list[str],
    dry_run: bool = False,
) -> dict:
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

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_message(place_name, reviews)},
                ],
                temperature=0.1,
            )
            raw    = clean_response(response.choices[0].message.content)
            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse hatası (deneme {attempt}/{MAX_RETRIES}): {e}")
        except Exception as e:
            log.warning(f"  Hata (deneme {attempt}/{MAX_RETRIES}): {e}")

        if attempt == MAX_RETRIES:
            raise RuntimeError(f"'{place_name}' için {MAX_RETRIES} denemede de analiz başarısız.")

        log.info(f"  Tekrar deneniyor...")

    raise RuntimeError(f"'{place_name}' analiz başarısız.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, output_path: str, model: str, dry_run: bool = False):
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

    client = None
    if not dry_run:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        log.info(f"Ollama bağlantısı: {OLLAMA_BASE_URL} | Model: {model}")

    results = []
    failed  = []

    for idx, entry in enumerate(scraped_data, start=1):
        place_name = extract_place_name(entry)
        reviews    = entry.get("reviews", [])

        log.info(f"[{idx}/{total}] '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning("  Yorum bulunamadı, atlanıyor.")
            failed.append({"place_name": place_name, "reason": "yorum yok"})
            continue

        try:
            result = analyze_single(client, model, place_name, reviews, dry_run=dry_run)

            result["place_name"]            = place_name
            result["source_url"]            = entry.get("url", "")
            result["total_reviews_scraped"] = entry.get("total_reviews_scraped", len(reviews))
            result["analyzed_at"]           = datetime.now(UTC).isoformat()

            results.append(result)
            log.info(f"  ✓ Genel puan: {result.get('genel_puan', '?')}")

        except RuntimeError as e:
            log.error(f"  ✗ {e}")
            failed.append({"place_name": place_name, "reason": str(e)})

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
    parser.add_argument("--input",   default="scraped_data.json")
    parser.add_argument("--output",  default="analyzed_data.json")
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="Ollama model adı")
    parser.add_argument("--dry-run", action="store_true", help="API çağrısı yapmadan test et")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        dry_run=args.dry_run,
    )