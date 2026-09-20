from datetime import datetime
from pydantic import BaseModel, Field


class RawNotification(BaseModel):
    id: str
    app_name: str
    sender: str
    title: str
    body: str
    timestamp: datetime


class CurrentContext(BaseModel):
    active_process: str
    window_title: str
    last_updated: datetime


class FilterLabel(BaseModel):
    urgency_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    category: str
    ai_summary_reason: str


class FilteringSample(BaseModel):
    notification: RawNotification
    context: CurrentContext
    label: FilterLabel