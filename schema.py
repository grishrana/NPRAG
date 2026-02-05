from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question (Nepali/English/Romanized)")


class Source(BaseModel):
    point_id: str
    score: float
    doc_id: Optional[str] = None
    source_path: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
