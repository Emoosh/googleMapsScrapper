"""
place_qa.py
-----------
RAG mikroservisi: ChromaDB'den ilgili yorumları çekip Gemini ile cevap üretir.

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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
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
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
EMBED_MODEL     = "gemini-embedding-001"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "place_reviews"
DEFAULT_TOP_K   = 12

# ---------------------------------------------------------------------------
# Singletons (uygulama başlarken bir kez init edilir)
# ---------------------------------------------------------------------------
gemini_client   = genai.Client(api_key=GEMINI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
chroma_client   = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
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

def build_prompt(place_name: str, question: str, reviews: list[str]) -> str:
    reviews_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    return f"""Aşağıdaki yorumlar '{place_name}' adlı mekana ait Google Maps yorumlarıdır.
Yalnızca bu yorumlara dayanarak kullanıcının sorusunu Türkçe olarak yanıtla.
Yorumlarda cevap yoksa "Bu konuda yorumlarda yeterli bilgi bulunamadı." de.
Kısa, net ve doğrudan cevap ver.

Soru: {question}

Yorumlar:
{reviews_text}
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "total_indexed_reviews": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1. Soruyu embed et
    embed_result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=req.question,
    )
    question_embedding = embed_result.embeddings[0].values

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

    # 3. Claude ile cevap üret
    message = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(req.place_name, req.question, reviews)}],
    )

    log.info(f"[/ask] '{req.place_name}' | soru: '{req.question}' | {len(reviews)} kaynak yorum")

    return AskResponse(
        place_name=req.place_name,
        question=req.question,
        answer=message.content[0].text.strip(),
        sources_used=len(reviews),
    )