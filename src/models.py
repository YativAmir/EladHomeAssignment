from typing import Optional
from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    text: str
    topic: str
    doc_title: str
    source: str
    page: int
    chunk_index: int
    section: Optional[str] = None
