from fastapi import FastAPI
from schema import ChatRequest, ChatResponse
from chat import chat as rag_chat
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app:FastAPI):
    load_dotenv()
    # --- Startup ---

    app.state.embedder = SentenceTransformer(
        os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    )

    app.state.collection = os.getenv("QDRANT_COLLECTION", "NPRAG")
    app.state.q_client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333")
    )
    print("✅ Lifespan startup complete")

    yield  # ← application runs here

    # --- Shutdown ---
    # (Optional cleanup if needed)
    print("🛑 Lifespan shutdown complete")

app = FastAPI(title="NPRAG", lifespan=lifespan)



@app.get("/")
def hello():
    return {"data": "Hello World"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    response = rag_chat(payload.question, q_client=app.state.q_client, embedder=app.state.embedder)
    answer = response.text

    return ChatResponse(answer=answer)
