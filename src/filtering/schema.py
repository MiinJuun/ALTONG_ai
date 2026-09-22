from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator


def validate_utc_z_datetime(value: object) -> object:
    if isinstance(value, str):
        if not value.endswith("Z"):
            raise ValueError("datetime must use UTC ISO 8601 format ending in Z")
        return value

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("datetime must use UTC")

    return value


class RawNotification(BaseModel):
    id: str
    app_name: str
    sender: str
    title: str
    body: str
    timestamp: datetime

    _validate_timestamp = field_validator("timestamp", mode="before")(
        validate_utc_z_datetime
    )


class CurrentContext(BaseModel):
    active_process: str
    window_title: str
    last_updated: datetime

    _validate_last_updated = field_validator("last_updated", mode="before")(
        validate_utc_z_datetime
    )


class FilterLabel(BaseModel):
    urgency_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    category: str
    ai_summary_reason: str


class FilteringSample(BaseModel):
    notification: RawNotification
    context: CurrentContext
    label: FilterLabel
