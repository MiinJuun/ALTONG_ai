from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.briefing.pipeline import SessionBriefingService


def notification(
    notification_id: str,
    *,
    title: str,
    body: str,
    timestamp: str = "2026-09-13T18:05:00Z",
) -> dict[str, object]:
    return {
        "id": notification_id,
        "app_name": "Slack",
        "sender": "가상 팀장",
        "title": title,
        "body": body,
        "timestamp": timestamp,
    }


def filter_result(
    notification_id: str,
    *,
    is_passed: bool = False,
) -> dict[str, object]:
    return {
        "notification_id": notification_id,
        "is_passed": is_passed,
        "urgency_score": 3,
        "relevance_score": 4,
        "category": "업무",
        "ai_summary_reason": "가상 테스트 판단",
    }


class ActionItemExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SessionBriefingService()
        self.generated_at = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)

    def build(
        self,
        notifications: list[dict[str, object]],
        results: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self.service.build(
            session_id="session_actions",
            notifications=notifications,
            filter_results=(
                results
                if results is not None
                else [filter_result(item["id"]) for item in notifications]
            ),
            generated_at=self.generated_at,
        ).to_dict()

    def test_empty_input_returns_empty_candidate_lists(self) -> None:
        briefing = self.build([])

        self.assertEqual(briefing["todo_candidate_count"], 0)
        self.assertEqual(briefing["todo_candidates"], [])
        self.assertEqual(briefing["calendar_candidate_count"], 0)
        self.assertEqual(briefing["calendar_candidates"], [])

    def test_passed_notification_is_not_extracted(self) -> None:
        item = notification(
            "n1",
            title="보고서 제출 요청",
            body="보고서를 내일까지 제출해 주세요.",
        )
        briefing = self.build([item], [filter_result("n1", is_passed=True)])

        self.assertEqual(briefing["todo_candidates"], [])
        self.assertEqual(briefing["calendar_candidates"], [])

    def test_todo_with_korean_due_date_is_structured(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="보고서 제출 요청",
                    body="보고서를 9월 15일까지 제출해 주세요.",
                )
            ]
        )

        candidate = briefing["todo_candidates"][0]
        self.assertEqual(candidate["due_at"], "2026-09-15T00:00:00Z")
        self.assertTrue(candidate["is_all_day"])
        self.assertEqual(candidate["source_notification_ids"], ["n1"])
        self.assertIn("제출", candidate["matched_cues"])

    def test_calendar_candidate_parses_explicit_date_and_time(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="프로젝트 회의 일정",
                    body="프로젝트 회의는 2026년 9월 20일 오후 3시에 진행됩니다.",
                )
            ]
        )

        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["scheduled_at"], "2026-09-20T15:00:00Z")
        self.assertFalse(candidate["is_all_day"])
        self.assertEqual(candidate["title"], "프로젝트 회의 일정")

    def test_relative_date_is_based_on_notification_timestamp(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="프로젝트 회의",
                    body="프로젝트 회의는 내일 오전 10시 30분에 진행됩니다.",
                )
            ]
        )

        self.assertEqual(
            briefing["calendar_candidates"][0]["scheduled_at"],
            "2026-09-14T10:30:00Z",
        )

    def test_repeated_candidates_are_merged_with_all_sources(self) -> None:
        items = [
            notification(
                "n1",
                title="과제 제출",
                body="과제를 9월 15일까지 제출해 주세요.",
                timestamp="2026-09-13T18:05:00Z",
            ),
            notification(
                "n2",
                title="과제 제출",
                body="과제를 9월 15일까지 제출해 주세요.",
                timestamp="2026-09-13T19:05:00Z",
            ),
        ]
        briefing = self.build(items)

        self.assertEqual(briefing["todo_candidate_count"], 1)
        self.assertEqual(briefing["calendar_candidate_count"], 1)
        self.assertEqual(
            briefing["todo_candidates"][0]["source_notification_ids"],
            ["n1", "n2"],
        )
        self.assertEqual(
            briefing["calendar_candidates"][0]["source_notification_ids"],
            ["n1", "n2"],
        )

    def test_invalid_calendar_date_is_not_emitted(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="팀 회의 일정",
                    body="팀 회의 일정은 2월 30일입니다.",
                )
            ]
        )

        self.assertEqual(briefing["calendar_candidates"], [])


if __name__ == "__main__":
    unittest.main()
