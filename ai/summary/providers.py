"""Provider boundaries and the deterministic MVP briefing provider."""

from __future__ import annotations

from typing import Protocol, Sequence

from .models import BriefingItem, RawNotification


class NotificationSanitizer(Protocol):
    """Future boundary for masking data immediately before an external call.

    The first MVP neither implements nor invokes this boundary because it uses
    synthetic data and never calls an external provider.
    """

    def sanitize(self, notification: RawNotification) -> RawNotification: ...


class BriefingProvider(Protocol):
    """Replaceable group-summary provider boundary."""

    def summarize(self, items: Sequence[BriefingItem]) -> tuple[str, ...]: ...


class RuleBasedBriefingProvider:
    """Create at most three factual lines without a model or network call."""

    def summarize(self, items: Sequence[BriefingItem]) -> tuple[str, ...]:
        if not items:
            return ()

        ordered = sorted(items, key=lambda item: item.notification.timestamp)
        first = ordered[0].notification
        titles = list(dict.fromkeys(item.notification.title for item in ordered))
        lines = [f"{first.app_name}의 {first.sender} 알림 {len(ordered)}건"]
        lines.append(f"주요 제목: {', '.join(titles[:2])}")
        if len(titles) > 2:
            lines.append(f"그 외 제목 {len(titles) - 2}건")
        else:
            latest = ordered[-1].notification.body.strip()
            lines.append(f"최근 내용: {latest[:80]}")
        return tuple(lines[:3])

