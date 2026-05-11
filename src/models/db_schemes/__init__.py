from pydantic import BaseModel
from enum import Enum
from typing import Optional


class LanguageEnum(str, Enum):
    EN = "en"
    AR = "ar"


class DataChunk(BaseModel):
    chunk_id: str
    collection_name: str
    text: str
    source_url: str
    source_name: str
    topic: str
    language: LanguageEnum = LanguageEnum.EN
    order: int
    tokens: int


class QueryResult(BaseModel):
    answer: str
    sources: list
    collection: str
    language: LanguageEnum = LanguageEnum.EN
