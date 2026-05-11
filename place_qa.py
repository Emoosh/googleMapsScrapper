"""
place_qa.py
-----------
RAG mikroservisi: ChromaDB'den ilgili yorumları çekip Claude Haiku ile cevap üretir.
Embedding local olarak intfloat/multilingual-e5-large modeli ile yapılır.

Önce indexer.py çalıştırılmalı!

Kullanım:
    uvicorn place_qa:app --reload --port 8001

Endpoint'ler:
    GET  /health
    POST /ask   {"place_name": "Fanus Komünite Kafe", "question": "Vegan seçenek var mı?"}
"""

import os
import re
import logging
import sys

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
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
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL   = "intfloat/multilingual-e5-large"
CHROMA_PATH   = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "place_reviews"
DEFAULT_TOP_K   = 6

SYSTEM_PROMPT = (
    "Sen bir mekan değerlendirme asistanısın. "
    "Yalnızca verilen Google Maps yorumlarına dayanarak Türkçe yanıtla. "
    "Yorumlarda bilgi yoksa tek cümleyle 'Bu konuda yorumlarda bilgi bulunamadı.' de. "
    "Yorumlarda bilgi varsa 2-3 cümleyle özetle. "
    "Asla düşünce sürecini yazma, doğrudan cevabı ver."
)

# CUDA > MPS > CPU otomatik seçim
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Singletons (uygulama başlarken bir kez init edilir)
# ---------------------------------------------------------------------------
log.info(f"Embedding modeli yükleniyor: {EMBED_MODEL} ({DEVICE})")
embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

app = FastAPI(title="Place Q&A Mikroservisi")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    place_name: str
    question:   str
    top_k:      int = DEFAULT_TOP_K


class AskResponse(BaseModel):
    place_name:   str
    question:     str
    answer:       str
    sources_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_message(place_name: str, question: str, reviews: list[str]) -> str:
    reviews_text = "\n".join(f"{i+1}. {r[:200]}" for i, r in enumerate(reviews))
    return f"Mekan: {place_name}\nSoru: {question}\n\nYorumlar:\n{reviews_text}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "collection": COLLECTION_NAME,
        "total_indexed_reviews": get_collection().count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    collection = get_collection()

    # 1. Soruyu local model ile embed et (multilingual-e5 için "query: " prefix'i)
    question_embedding = embed_model.encode(
        f"query: {req.question}",
        normalize_embeddings=True,
    ).tolist()

    # 2. ChromaDB'de o mekana ait en alakalı yorumları çek
    existing = collection.get(where={"place_name": req.place_name}, limit=1)
    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.place_name}' için indexli yorum bulunamadı.",
        )
    place_docs = collection.get(where={"place_name": req.place_name})
    n = min(req.top_k, len(place_docs["ids"]))
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n,
        where={"place_name": req.place_name},
    )

    reviews = results["documents"][0] if results["documents"] else []
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.place_name}' için indexli yorum bulunamadı. Önce indexer.py çalıştırın.",
        )

    # 3. Claude Haiku ile cevap üret
    response = llm_client.messages.create(
        model=LLM_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(req.place_name, req.question, reviews)}],
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.content[0].text or ""
    if "</think>" in raw:
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        answer = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    log.info(f"[/ask] '{req.place_name}' | soru: '{req.question}' | {len(reviews)} kaynak yorum")

    return AskResponse(
        place_name=req.place_name,
        question=req.question,
        answer=answer,
        sources_used=len(reviews),
    )