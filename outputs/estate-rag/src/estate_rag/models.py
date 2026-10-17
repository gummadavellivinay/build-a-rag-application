from typing import Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    page: int = 1
    tenant: str
    listing_id: str = ""
    language: str = "und"
    modality: Literal["text", "ocr"] = "text"
    ocr_confidence: float | None = None


class Evidence(BaseModel):
    id: str
    text: str
    source: str
    page: int | None = None
    kind: Literal["document", "graph", "tool"] = "document"
    score: float | None = None


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["hybrid", "multimodal", "graph"] = "hybrid"


class Plan(BaseModel):
    entity_mentions: list[str] = Field(default_factory=list, max_length=5)
    needs_live_status: bool = False


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Draft(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    insufficient_evidence: bool = False


class Verdict(BaseModel):
    supported: bool
    reason: str


class Answer(BaseModel):
    answer: str
    evidence: list[Evidence]
    abstained: bool
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
