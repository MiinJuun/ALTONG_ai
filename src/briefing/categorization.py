"""Rule-based group categorization used as an MVP comparison baseline."""

from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

from .schema import BriefingItem, CategoryDecision


WORK = "업무"
SCHEDULE = "일정"
CHAT = "잡담"

_CATEGORY_HINTS = {
    SCHEDULE: ("일정", "캘린더", "회의", "약속", "마감", "기한", "예약"),
    CHAT: ("잡담", "일상", "친구", "단톡", "광고", "홍보", "소셜"),
    WORK: ("업무", "긴급", "개발", "학업", "과제", "프로젝트", "공지"),
}
_DATE_OR_TIME_PATTERN = re.compile(
    r"(?:\d{1,2}[월/.]\s*\d{1,2}일?|\d{1,2}:\d{2}|오늘|내일|모레|회의|일정|마감|기한)"
)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def canonical_category(item: BriefingItem) -> str:
    """Map an item to one of the temporary README-level categories.

    The existing filter category is considered first.  Text hints are only a
    deterministic fallback for categories not covered by the temporary map.
    """

    category = _normalized(item.filter_result.category)
    for canonical in (SCHEDULE, CHAT, WORK):
        if any(hint in category for hint in _CATEGORY_HINTS[canonical]):
            return canonical

    notification = item.notification
    text = _normalized(f"{notification.title} {notification.body}")
    if _DATE_OR_TIME_PATTERN.search(text):
        return SCHEDULE
    if any(hint in text for hint in _CATEGORY_HINTS[CHAT]):
        return CHAT
    return WORK


def categorize_group(items: list[BriefingItem]) -> CategoryDecision:
    """Choose a primary category after grouping all related notifications.

    Each notification has one vote.  Ties prefer the category of the latest
    notification so a later update can influence the final state.  Urgency and
    relevance remain separate signals and never redefine semantic category.
    """

    if not items:
        raise ValueError("cannot categorize an empty group")

    votes: dict[str, int] = defaultdict(int)
    evidence: dict[str, list[str]] = defaultdict(list)
    ordered = sorted(items, key=lambda item: item.notification.timestamp)

    for item in ordered:
        category = canonical_category(item)
        votes[category] += 1
        evidence[category].append(item.notification.id)

    highest_score = max(votes.values())
    tied = {category for category, score in votes.items() if score == highest_score}
    latest_category = canonical_category(ordered[-1])
    primary = latest_category if latest_category in tied else sorted(tied)[0]

    return CategoryDecision(
        primary_category=primary,
        evidence_notification_ids=tuple(evidence[primary]),
    )
