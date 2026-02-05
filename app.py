from fastapi import FastAPI
from schema import ChatRequest, ChatResponse
from chat import chat as rag_chat

app = FastAPI(title="NPRAG")

@app.get("/")
def hello():
    return {"data": "Hello World"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    response = rag_chat(payload.question)
    answer = response.text

    return ChatResponse(answer=answer)
