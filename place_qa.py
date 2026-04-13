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
import logging
import sys

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
EMBED_MODEL       = "intfloat/multilingual-e5-large"
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "place_reviews"
DEFAULT_TOP_K     = 6

SYSTEM_PROMPT = (
    "Sen bir mekan değerlendirme asistanısın. "
    "Yalnızca sana verilen Google Maps yorumlarına dayanarak soruları Türkçe yanıtla. "
    "Yorumlarda cevap yoksa 'Bu konuda yorumlarda yeterli bilgi bulunamadı.' de. "
    "Kısa, net ve doğrudan cevap ver."
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
embed_model      = SentenceTransformer(EMBED_MODEL, device=DEVICE)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
chroma_client    = chromadb.PersistentClient(path=CHROMA_PATH)
collection       = chroma_client.get_or_create_collection(
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
        "total_indexed_reviews": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1. Soruyu local model ile embed et (multilingual-e5 için "query: " prefix'i)
    question_embedding = embed_model.encode(
        f"query: {req.question}",
        normalize_embeddings=True,
    ).tolist()

    # 2. ChromaDB'de o mekana ait en alakalı yorumları çek
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=req.top_k,
        where={"place_name": req.place_name},
    )

    reviews = results["documents"][0] if results["documents"] else []
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.place_name}' için indexli yorum bulunamadı. Önce indexer.py çalıştırın.",
        )

    # 3. Claude Haiku ile cevap üret
    message = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(req.place_name, req.question, reviews)}],
    )

    answer = message.content[0].text.strip()
    log.info(f"[/ask] '{req.place_name}' | soru: '{req.question}' | {len(reviews)} kaynak yorum")

    return AskResponse(
        place_name=req.place_name,
        question=req.question,
        answer=answer,
        sources_used=len(reviews),
    )