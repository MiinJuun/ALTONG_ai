import json
import unittest

from pydantic import ValidationError

from src.filtering.prompt import build_messages
from src.filtering.schema import CurrentContext, RawNotification


BASE_CONTEXT = {
    "active_process": "Code.exe",
    "window_title": "auth.py - Visual Studio Code",
    "last_updated": "2026-09-21T08:59:55Z",
}


class ContextExtensionTests(unittest.TestCase):
    def test_legacy_and_empty_context_use_consistent_defaults(self) -> None:
        legacy = CurrentContext.model_validate(BASE_CONTEXT)
        empty = CurrentContext.model_validate({
            "ActiveProcess": "",
            "WindowTitle": "",
            "LastUpdated": "2026-09-21T08:59:55Z",
            "DurationSeconds": 0,
            "RecentProcesses": None,
        })
        self.assertEqual(legacy.duration_seconds, 0)
        self.assertEqual(legacy.recent_processes, [])
        self.assertEqual(empty.recent_processes, [])

    def test_pascal_and_camel_case_wire_data_normalize_to_snake_case_prompt(self) -> None:
        notification = RawNotification.model_validate({
            "Id": "noti_test",
            "AppName": "Slack",
            "Sender": "가상 발신자",
            "Title": "검토 요청",
            "Body": "인증 API 검토 요청",
            "Timestamp": "2026-09-21T09:00:00Z",
        })
        context = CurrentContext.model_validate({
            "activeProcess": "Code.exe",
            "windowTitle": "auth.py - Visual Studio Code",
            "lastUpdated": "2026-09-21T08:59:55Z",
            "durationSeconds": 42,
            "recentProcesses": ["Code.exe", "chrome.exe", "WindowsTerminal.exe"],
        })
        payload = json.loads(build_messages(notification, context)[1]["content"])
        self.assertEqual(payload["notification"]["app_name"], "Slack")
        self.assertEqual(payload["context"]["duration_seconds"], 42)
        self.assertEqual(payload["context"]["recent_processes"], [
            "Code.exe", "chrome.exe", "WindowsTerminal.exe"
        ])
        self.assertEqual(payload["context"]["last_updated"], "2026-09-21T08:59:55Z")

    def test_invalid_recent_processes_and_duration_are_rejected(self) -> None:
        invalid = [
            {"duration_seconds": -1},
            {"duration_seconds": "5"},
            {"recent_processes": ["a", "b", "c", "d"]},
            {"recent_processes": ["Code.exe", " "]},
        ]
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                CurrentContext.model_validate({**BASE_CONTEXT, **extra})


if __name__ == "__main__":
    unittest.main()
