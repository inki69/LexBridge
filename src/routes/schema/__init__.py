from pydantic import BaseModel, Field
from typing import List, Optional
from models.db_schemes import LanguageEnum


class QueryRequest(BaseModel):
    text: str
    collection: str = "lexbridge"
    k: int = Field(default=5, ge=1, le=20)
    language: LanguageEnum = LanguageEnum.EN


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_name: str
    source_url: str
    topic: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    collection: str
    model: str


class UploadResponse(BaseModel):
    message: str
    chunks_created: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    db_connected: bool
    vector_db_ready: bool


class CollectionInfoResponse(BaseModel):
    collection: str
    exists: bool
    points_count: int
    dimension: Optional[int] = None
