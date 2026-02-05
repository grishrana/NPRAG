from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_COLLECTION = os.getenv('QDRANG_COLLECTION', "NPRAG")
MODEL = "gemini-2.5-flash"

EMBED_MODEL="BAAI/bge-m3"
VECTOR_SIZE=1024
TOP_K=5

# clietns

q_client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)
gem_client = genai.Client(api_key=GEMINI_API_KEY)

def retrieve(query: str, top_k: int = TOP_K):
    query_vec= embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    results = q_client.query_points(collection_name=QDRANT_COLLECTION,
                             query=query_vec.tolist(),
                             limit=top_k,
                             with_payload=True)
    contexts = []
    for res in results.points:
        payload = res.payload or {}
        text = payload.get("text")
        if text:
            contexts.append(text)
    return contexts

def build_prompt(question: str, contexts: List[str]):
    context_block = "\n\n---\n\n".join(contexts)
    return f"""
    You are a factual question-answering assistant.

    Rules:
    - Answer ONLY using the provided context.
    - If the answer is not present in the context, say you do not know.
    - Respond in the SAME language/script as the user:
    - Nepali → Nepali
    - English → English
    - Romanized Nepali → Romanized Nepali
    - Be concise and correct.

    Context:
    {context_block}

    Question:
    {question}

    Answer:
    """.strip()

def chat(prompt: str):
    contexts = retrieve(prompt)
    if not contexts:
        return "माफ गर्नुहोस्, उपलब्ध सन्दर्भमा उत्तर फेला परेन।"

    prompt = build_prompt(prompt, contexts)

    response = gem_client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("\nAnswer:\n", chat(q))
