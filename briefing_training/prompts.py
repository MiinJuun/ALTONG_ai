"""Prompt and response contract for the local Qwen briefing baseline."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_SUMMARY_LINES = 3
MAX_LINE_LENGTH = 160

SYSTEM_PROMPT = """당신은 PC 집중 세션이 끝난 뒤 차단된 알림을 요약하는 도우미입니다.

반드시 다음 규칙을 지키세요.
1. 입력 알림에 명시된 사실만 사용하고 추측하거나 새로운 사실을 만들지 마세요.
2. 동일한 사건의 반복 알림은 하나의 흐름으로 합치세요.
3. 변경, 취소, 복구처럼 상태가 달라졌다면 가장 최신 알림의 상태를 우선하세요.
4. 긴급도와 연관도가 높은 내용을 먼저 쓰세요.
5. 제목만 나열하지 말고 본문에 있는 핵심 사실을 포함하세요.
6. 일정과 제출 알림은 날짜, 시간, 제출물 등 사용자가 행동하는 데 필요한 정보를 보존하세요.
7. 각 줄은 알림 표시나 필드 이름 없이 그 자체로 이해되는 완전한 문장이어야 합니다.
8. 한국어로 짧게 작성하고 요약은 최대 3줄까지만 만드세요.
9. 마크다운, 설명, 사고 과정 없이 아래 JSON 객체 하나만 출력하세요.
10. 최상위 값은 배열이 아니라 반드시 summary_lines 필드가 있는 객체여야 합니다.

출력 스키마:
{"summary_lines":["첫 번째 요약", "두 번째 요약"]}

잘못된 출력:
["첫 번째 요약", "두 번째 요약"]

올바른 출력:
{"summary_lines":["첫 번째 요약", "두 번째 요약"]}
"""


class SummaryResponseError(ValueError):
    """Raised when a model response violates the summary JSON contract."""


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _score_text(value: object) -> str:
    if isinstance(value, Mapping):
        minimum = value.get("min")
        maximum = value.get("max")
        average = value.get("average")
        return f"min={minimum}, max={maximum}, average={average}"
    return str(value)


def build_messages(group: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build Qwen chat messages from one already-grouped notification unit."""

    notifications = group.get("notifications")
    if not isinstance(notifications, Sequence) or isinstance(notifications, (str, bytes)):
        raise ValueError("notifications must be a non-empty sequence")
    if not notifications:
        raise ValueError("notifications must not be empty")

    notification_records: list[dict[str, str]] = []
    for notification in notifications:
        if not isinstance(notification, Mapping):
            raise ValueError("each notification must be an object")
        notification_records.append(
            {
                "id": _required_text(notification, "id"),
                "timestamp": _required_text(notification, "timestamp"),
                "title": _required_text(notification, "title"),
                "body": _required_text(notification, "body"),
            }
        )

    group_context = {
        "app_name": _required_text(group, "app_name"),
        "sender": _required_text(group, "sender"),
        "category": _required_text(group, "category"),
        "urgency_score": _score_text(group.get("urgency_score")),
        "relevance_score": _score_text(group.get("relevance_score")),
        "notifications_oldest_to_newest": notification_records,
    }
    user_prompt = "\n".join(
        (
            "다음 알림 그룹을 요약하세요.",
            "",
            "입력 JSON:",
            json.dumps(group_context, ensure_ascii=False, separators=(",", ":")),
            "",
            "제목만 복사하지 말고 body의 구체적인 핵심 사실을 포함하세요.",
            "최신 알림이 취소, 변경, 복구를 알리면 그 최종 상태를 분명히 쓰세요.",
            "입력의 id, 필드 이름, '알림 1' 같은 표시는 출력하지 마세요.",
            "반드시 { 문자로 시작하고 } 문자로 끝나는 JSON 객체만 출력하세요.",
            '형식: {"summary_lines":["요약 문장"]}',
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_summary_response(raw_response: str) -> tuple[str, ...]:
    """Parse and strictly validate the JSON returned by the model."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise SummaryResponseError("model response must be a non-empty string")

    candidate = raw_response.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3:
            raise SummaryResponseError("invalid fenced JSON response")
        candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SummaryResponseError("model response is not valid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"summary_lines"}:
        raise SummaryResponseError(
            "model response must contain only the summary_lines field"
        )

    summary_lines = payload["summary_lines"]
    if not isinstance(summary_lines, list):
        raise SummaryResponseError("summary_lines must be an array")
    if not 1 <= len(summary_lines) <= MAX_SUMMARY_LINES:
        raise SummaryResponseError("summary_lines must contain one to three lines")

    normalized: list[str] = []
    for line in summary_lines:
        if not isinstance(line, str) or not line.strip():
            raise SummaryResponseError("each summary line must be a non-empty string")
        text = line.strip()
        if len(text) > MAX_LINE_LENGTH:
            raise SummaryResponseError(
                f"each summary line must be at most {MAX_LINE_LENGTH} characters"
            )
        normalized.append(text)
    return tuple(normalized)
