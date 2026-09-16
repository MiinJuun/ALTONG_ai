"""Deterministic baseline grouping for blocked notifications."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import re
import unicodedata

from .models import BriefingItem


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")
_STOP_WORDS = {
    "관련",
    "대한",
    "부탁",
    "확인",
    "주세요",
    "합니다",
    "해주세요",
    "지금",
    "the",
    "and",
    "for",
    "with",
}


def normalize_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def text_tokens(item: BriefingItem) -> set[str]:
    text = unicodedata.normalize(
        "NFKC", f"{item.notification.title} {item.notification.body}"
    ).casefold()
    return {
        token
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token not in _STOP_WORDS
    }


def hour_bucket(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


@dataclass(slots=True)
class RuleGroup:
    app_key: str
    sender_key: str
    bucket: datetime
    items: list[BriefingItem] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)

    def accepts(self, item: BriefingItem, tokens: set[str]) -> bool:
        notification = item.notification
        if normalize_identity(notification.app_name) != self.app_key:
            return False
        if normalize_identity(notification.sender) != self.sender_key:
            return False
        if hour_bucket(notification.timestamp) != self.bucket:
            return False
        return bool(self.tokens & tokens)

    def add(self, item: BriefingItem, tokens: set[str]) -> None:
        self.items.append(item)
        self.tokens.update(tokens)


def group_items(items: list[BriefingItem]) -> list[RuleGroup]:
    """Group by app, sender, UTC hour, and at least one shared text token."""

    groups: list[RuleGroup] = []
    ordered = sorted(
        items,
        key=lambda item: (item.notification.timestamp, item.notification.id),
    )

    for item in ordered:
        tokens = text_tokens(item)
        matching = next((group for group in groups if group.accepts(item, tokens)), None)
        if matching is None:
            notification = item.notification
            matching = RuleGroup(
                app_key=normalize_identity(notification.app_name),
                sender_key=normalize_identity(notification.sender),
                bucket=hour_bucket(notification.timestamp),
            )
            groups.append(matching)
        matching.add(item, tokens)

    return groups


def representative_keywords(group: RuleGroup, limit: int = 3) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for item in group.items:
        counts.update(text_tokens(item))
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(token for token, _ in ranked[:limit])
