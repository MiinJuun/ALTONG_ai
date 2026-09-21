"""Deterministic To-Do and calendar candidate extraction for the MVP.

The rules deliberately favor traceable candidates over aggressive extraction.
This provider can later be replaced by an LLM-backed implementation without
changing the session briefing contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
from typing import Protocol, Sequence
import unicodedata

from .schema import BriefingItem, CalendarCandidate, TodoCandidate


_TODO_CUES = (
    "해 주세요",
    "해주세요",
    "부탁",
    "필요",
    "해야",
    "제출",
    "확인",
    "수정",
    "작성",
    "전달",
    "완료",
    "재시도",
    "등록",
)
_CALENDAR_CUES = (
    "회의",
    "일정",
    "마감",
    "기한",
    "시험",
    "발표",
    "면담",
    "약속",
    "예약",
    "수업",
    "세미나",
    "행사",
    "제출",
)
_RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2}
_DATE_PATTERN = re.compile(
    r"(?:(?P<year>\d{4})\s*(?:년|[./-])\s*)?"
    r"(?P<month>\d{1,2})\s*(?:월|[./-])\s*"
    r"(?P<day>\d{1,2})\s*일?"
)
_AMPM_TIME_PATTERN = re.compile(
    r"(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2})\s*시"
    r"(?:\s*(?P<minute>[0-5]?\d)\s*분)?"
)
_COLON_TIME_PATTERN = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?!\d)"
)
_HOUR_TIME_PATTERN = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3])\s*시"
    r"(?:\s*(?P<minute>[0-5]?\d)\s*분)?"
)


@dataclass(frozen=True, slots=True)
class CandidateExtraction:
    todos: tuple[TodoCandidate, ...] = ()
    calendar: tuple[CalendarCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class _TemporalMatch:
    value: datetime
    is_all_day: bool


class ActionItemProvider(Protocol):
    """Replaceable boundary for To-Do and calendar candidate extraction."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> CandidateExtraction: ...


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _candidate_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{digest}"


def _display_text(item: BriefingItem) -> str:
    notification = item.notification
    text = f"{notification.title} — {notification.body}".strip()
    if len(text) > 240:
        return f"{text[:237].rstrip()}..."
    return text


def _matched_cues(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(cue for cue in cues if cue in normalized)


def _parse_time(text: str) -> tuple[int, int] | None:
    ampm_match = _AMPM_TIME_PATTERN.search(text)
    if ampm_match:
        hour = int(ampm_match.group("hour"))
        if not 1 <= hour <= 12:
            return None
        if ampm_match.group("ampm") == "오전":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        return hour, int(ampm_match.group("minute") or 0)

    colon_match = _COLON_TIME_PATTERN.search(text)
    if colon_match:
        return int(colon_match.group("hour")), int(colon_match.group("minute"))

    hour_match = _HOUR_TIME_PATTERN.search(text)
    if hour_match:
        return int(hour_match.group("hour")), int(hour_match.group("minute") or 0)
    return None


def _parse_temporal(text: str, base: datetime) -> _TemporalMatch | None:
    base_utc = base.astimezone(timezone.utc)
    date_match = _DATE_PATTERN.search(text)
    relative_match = next((word for word in _RELATIVE_DAYS if word in text), None)
    parsed_time = _parse_time(text)

    if date_match:
        year = int(date_match.group("year") or base_utc.year)
        month = int(date_match.group("month"))
        day = int(date_match.group("day"))
    elif relative_match:
        relative_date = base_utc.date() + timedelta(days=_RELATIVE_DAYS[relative_match])
        year, month, day = relative_date.year, relative_date.month, relative_date.day
    elif parsed_time:
        year, month, day = base_utc.year, base_utc.month, base_utc.day
    else:
        return None

    hour, minute = parsed_time or (0, 0)
    try:
        value = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None
    return _TemporalMatch(value=value, is_all_day=parsed_time is None)


class RuleBasedActionItemProvider:
    """Extract conservative, source-grounded candidates from blocked items."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> CandidateExtraction:
        todos: list[TodoCandidate] = []
        calendar: list[CalendarCandidate] = []

        for item in items:
            notification = item.notification
            source_text = f"{notification.title} {notification.body}"
            temporal = _parse_temporal(source_text, notification.timestamp)
            todo_cues = _matched_cues(source_text, _TODO_CUES)
            calendar_cues = _matched_cues(source_text, _CALENDAR_CUES)

            if todo_cues:
                text = _display_text(item)
                temporal_key = temporal.value.isoformat() if temporal else "no-due-date"
                key = f"{_normalize(text)}|{temporal_key}"
                todos.append(
                    TodoCandidate(
                        candidate_id=_candidate_id("todo", key),
                        text=text,
                        due_at=temporal.value if temporal else None,
                        is_all_day=temporal.is_all_day if temporal else False,
                        source_notification_ids=(notification.id,),
                        source_group_ids=(group_id,),
                        matched_cues=todo_cues,
                    )
                )

            if calendar_cues and temporal:
                key = f"{_normalize(notification.title)}|{temporal.value.isoformat()}"
                calendar.append(
                    CalendarCandidate(
                        candidate_id=_candidate_id("calendar", key),
                        title=notification.title,
                        scheduled_at=temporal.value,
                        is_all_day=temporal.is_all_day,
                        source_notification_ids=(notification.id,),
                        source_group_ids=(group_id,),
                        matched_cues=calendar_cues,
                    )
                )

        return CandidateExtraction(todos=tuple(todos), calendar=tuple(calendar))


def merge_candidates(
    extractions: Sequence[CandidateExtraction],
) -> CandidateExtraction:
    """Merge repeated candidates while preserving every source reference."""

    todos: dict[str, TodoCandidate] = {}
    calendar: dict[str, CalendarCandidate] = {}

    for extraction in extractions:
        for candidate in extraction.todos:
            existing = todos.get(candidate.candidate_id)
            if existing is None:
                todos[candidate.candidate_id] = candidate
                continue
            todos[candidate.candidate_id] = TodoCandidate(
                candidate_id=existing.candidate_id,
                text=existing.text,
                due_at=existing.due_at or candidate.due_at,
                is_all_day=existing.is_all_day or candidate.is_all_day,
                source_notification_ids=tuple(
                    sorted(
                        set(
                            existing.source_notification_ids
                            + candidate.source_notification_ids
                        )
                    )
                ),
                source_group_ids=tuple(
                    sorted(set(existing.source_group_ids + candidate.source_group_ids))
                ),
                matched_cues=tuple(
                    sorted(set(existing.matched_cues + candidate.matched_cues))
                ),
            )

        for candidate in extraction.calendar:
            existing = calendar.get(candidate.candidate_id)
            if existing is None:
                calendar[candidate.candidate_id] = candidate
                continue
            calendar[candidate.candidate_id] = CalendarCandidate(
                candidate_id=existing.candidate_id,
                title=existing.title,
                scheduled_at=existing.scheduled_at,
                is_all_day=existing.is_all_day,
                source_notification_ids=tuple(
                    sorted(
                        set(
                            existing.source_notification_ids
                            + candidate.source_notification_ids
                        )
                    )
                ),
                source_group_ids=tuple(
                    sorted(set(existing.source_group_ids + candidate.source_group_ids))
                ),
                matched_cues=tuple(
                    sorted(set(existing.matched_cues + candidate.matched_cues))
                ),
            )

    todo_values = tuple(
        sorted(
            todos.values(),
            key=lambda candidate: (
                candidate.due_at is None,
                candidate.due_at or datetime.max.replace(tzinfo=timezone.utc),
                candidate.text,
            ),
        )
    )
    calendar_values = tuple(
        sorted(
            calendar.values(),
            key=lambda candidate: (candidate.scheduled_at, candidate.title),
        )
    )
    return CandidateExtraction(todos=todo_values, calendar=calendar_values)
