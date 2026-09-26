import json
import unittest

from src.filtering.prompt import build_messages, parse_model_output
from src.filtering.schema import CurrentContext, RawNotification


class PromptTests(unittest.TestCase):
    def test_prompt_contains_contract_input_and_no_label(self) -> None:
        notification = RawNotification(
            id="noti_test",
            app_name="TestApp",
            sender="가상 발신자",
            title="서버 장애",
            body="로그인 실패",
            timestamp="2026-09-21T09:00:00Z",
        )
        context = CurrentContext(
            active_process="Code.exe",
            window_title="auth.py - VS Code",
            last_updated="2026-09-21T08:59:55Z",
        )
        messages = build_messages(notification, context)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["notification"]["id"], "noti_test")
        self.assertEqual(payload["notification"]["timestamp"], "2026-09-21T09:00:00Z")
        self.assertEqual(payload["context"]["last_updated"], "2026-09-21T08:59:55Z")
        self.assertNotIn("label", payload)

    def test_valid_json_label(self) -> None:
        output = json.dumps({
            "urgency_score": 5,
            "relevance_score": 4,
            "category": "긴급 업무",
            "ai_summary_reason": "서버 장애로 즉시 확인이 필요함",
        })
        self.assertEqual(parse_model_output(output).urgency_score, 5)

    def test_rejects_invalid_output(self) -> None:
        valid = {
            "urgency_score": 5,
            "relevance_score": 4,
            "category": "긴급 업무",
            "ai_summary_reason": "확인이 필요함",
        }
        invalid = [
            "```json\n" + json.dumps(valid) + "\n```",
            json.dumps({**valid, "is_passed": True}),
            json.dumps({**valid, "urgency_score": "5"}),
            json.dumps({**valid, "relevance_score": 6}),
            json.dumps({**valid, "category": " "}),
            json.dumps({**valid, "category": "urgent"}),
        ]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_model_output(output)
