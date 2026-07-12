import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import client_usage_export
import monitor


class UsageHistoryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_history_path = monitor.USAGE_HISTORY_JSON
        monitor.USAGE_HISTORY_JSON = Path(self.temporary_directory.name) / "usage_history.json"
        self.day = monitor.today_key()

    def tearDown(self) -> None:
        monitor.USAGE_HISTORY_JSON = self.original_history_path
        self.temporary_directory.cleanup()

    def seed_history(self, source: str = "both") -> None:
        monitor.write_json_atomic(
            monitor.USAGE_HISTORY_JSON,
            {
                "schema": 1,
                "days": {
                    self.day: {
                        "date": self.day,
                        "source": source,
                        "requests": 100,
                        "tokens": 1_000_000,
                        "cost": 10.0,
                        "source_date": self.day,
                    }
                },
            },
        )

    def test_combined_usage_accepts_service_day_reset(self) -> None:
        self.seed_history("both")
        state = monitor.MonitorState(
            usage_source="both",
            today_requests=10,
            today_tokens=100_000,
            today_account_cost=1.0,
            client_usage={"date": self.day, "providers": []},
        )

        monitor.update_usage_history(state)

        saved = monitor.load_usage_history()["days"][self.day]
        self.assertEqual(state.today_tokens, 100_000)
        self.assertEqual(saved["tokens"], 100_000)

    def test_local_history_high_water_never_mutates_live_state(self) -> None:
        self.seed_history("local")
        state = monitor.MonitorState(
            usage_source="local",
            today_requests=10,
            today_tokens=100_000,
            today_account_cost=1.0,
            client_usage={"date": self.day, "providers": []},
        )

        monitor.update_usage_history(state)

        saved = monitor.load_usage_history()["days"][self.day]
        self.assertEqual(state.today_tokens, 100_000)
        self.assertEqual(saved["tokens"], 1_000_000)


class AccountUsageSortTests(unittest.TestCase):
    def test_5h_and_7d_sort_recently_used_accounts_first(self) -> None:
        rows = [
            {
                "name": "old-heavy",
                "tokens": 20_000_000,
                "requests": 50,
                "latest_at": "2026-06-25T10:00:00+08:00",
            },
            {
                "name": "current-light",
                "tokens": 1_000,
                "requests": 1,
                "latest_at": "2026-06-25T11:00:00+08:00",
            },
        ]

        ordered_5h = sorted(rows, key=lambda row: monitor.account_usage_sort_key(row, "5h"))
        ordered_7d = sorted(rows, key=lambda row: monitor.account_usage_sort_key(row, "7d"))

        self.assertEqual(ordered_5h[0]["name"], "current-light")
        self.assertEqual(ordered_7d[0]["name"], "current-light")

    def test_today_and_30d_sort_by_token_usage(self) -> None:
        rows = [
            {
                "name": "recent-light",
                "tokens": 1_000,
                "requests": 100,
                "latest_at": "2026-06-25T11:00:00+08:00",
            },
            {
                "name": "old-heavy",
                "tokens": 20_000_000,
                "requests": 1,
                "latest_at": "2026-06-25T10:00:00+08:00",
            },
        ]

        ordered_today = sorted(rows, key=lambda row: monitor.account_usage_sort_key(row, "today"))
        ordered_30d = sorted(rows, key=lambda row: monitor.account_usage_sort_key(row, "30d"))

        self.assertEqual(ordered_today[0]["name"], "old-heavy")
        self.assertEqual(ordered_30d[0]["name"], "old-heavy")


class ApiServicePoolAggregateTests(unittest.TestCase):
    def test_api_service_pool_row_sums_pool_accounts(self) -> None:
        rows = [
            {
                "name": "tissue",
                "tokens": 700,
                "requests": 7,
                "cost": 0.7,
                "latest_at": "2026-06-29T09:20:00+08:00",
                "latest_model": "gpt-5.4",
                "window_5h": {
                    "tokens": 650,
                    "requests": 6,
                    "cost": 0.65,
                    "remaining_percent": 99.0,
                    "utilization": 1.0,
                    "quota_available": True,
                    "latest_at": "2026-06-29T09:20:00+08:00",
                },
            },
            {
                "name": "hails",
                "tokens": 300,
                "requests": 3,
                "cost": 0.3,
                "latest_at": "2026-06-29T09:23:00+08:00",
                "latest_model": "gpt-5.5",
                "window_5h": {
                    "tokens": 300,
                    "requests": 3,
                    "cost": 0.3,
                    "remaining_percent": 98.0,
                    "utilization": 2.0,
                    "quota_available": True,
                    "latest_at": "2026-06-29T09:23:00+08:00",
                },
            },
        ]

        aggregate = monitor.build_api_service_pool_row(rows)

        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate["tokens"], 1000)
        self.assertEqual(aggregate["requests"], 10)
        self.assertAlmostEqual(aggregate["cost"], 1.0)
        self.assertEqual(aggregate["latest_at"], "2026-06-29T09:23:00+08:00")
        self.assertEqual(aggregate["latest_model"], "gpt-5.5")
        self.assertEqual(aggregate["window_5h"]["tokens"], 950)
        self.assertNotIn("quota_available", aggregate["window_5h"])
        self.assertNotIn("remaining_percent", aggregate["window_5h"])
        self.assertNotIn("utilization", aggregate["window_5h"])

    def test_api_service_local_mirror_is_subtracted_from_client_usage(self) -> None:
        usage = {
            "requests": 11,
            "tokens": 1100,
            "cost": 1.1,
            "providers": [
                {
                    "name": "Codex local - api-service-local",
                    "requests": 10,
                    "tokens": 1000,
                    "cost": 1.0,
                },
                {
                    "name": "Codex local - direct-account",
                    "requests": 1,
                    "tokens": 100,
                    "cost": 0.1,
                },
            ],
        }

        result = monitor.subtract_sub2api_mirrored_api_key_usage(usage, 1000, {})

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["tokens"], 100)
        self.assertEqual(len(result["providers"]), 1)
        self.assertEqual(result["providers"][0]["name"], "Codex local - direct-account")

    def test_account_row_pool_filter_uses_manifest_emails(self) -> None:
        pool = {"hails24.uranium@icloud.com", "tissue_wisp.24+g5@icloud.com"}

        self.assertTrue(
            monitor.account_row_matches_pool(
                {"name": "Codex local - hails24.uranium@icloud.com"},
                pool,
            )
        )
        self.assertFalse(
            monitor.account_row_matches_pool(
                {"name": "Codex local - rollers_tubers4s@icloud.com"},
                pool,
            )
        )


class LocalActiveAccountTests(unittest.TestCase):
    def test_active_accounts_are_deduped_by_active_sessions(self) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        usage = {
            "active_sessions": [
                {
                    "session_id": "session-1",
                    "provider": "Codex local - hails24.uranium@icloud.com",
                    "model": "gpt-5.5",
                    "latest_at": now,
                }
            ],
            "providers": [
                {
                    "name": "Codex local - hails24.uranium@icloud.com",
                    "latest_at": now,
                    "latest_model": "gpt-5.5",
                    "recent_sessions": 0,
                },
                {
                    "name": "Codex local - api-service-local",
                    "latest_at": now,
                    "latest_model": "gpt-5.5",
                    "recent_sessions": 0,
                },
                {
                    "name": "Codex local - codex_local_access_runtime",
                    "latest_at": now,
                    "latest_model": "gpt-5.5",
                    "recent_sessions": 0,
                },
            ],
        }

        active = monitor.local_active_accounts_from_client_usage(usage)

        self.assertEqual(len(active), 1)
        self.assertIn("hails24.uranium@icloud.com", active[0]["name"])
        self.assertEqual(active[0]["current"], 1)

    def test_recent_provider_without_recent_session_is_not_active(self) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        usage = {
            "providers": [
                {
                    "name": "Codex local - stale-provider",
                    "latest_at": now,
                    "latest_model": "gpt-5.5",
                    "recent_sessions": 0,
                }
            ],
            "latest_request": {},
        }

        active = monitor.local_active_accounts_from_client_usage(usage)

        self.assertEqual(active, [])


class LocalExportHighWaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temporary_directory.name) / "client_usage_today.json"
        self.day = date.today()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def snapshot(self, snapshot_day: date, tokens: int) -> dict:
        return {
            "date": snapshot_day.isoformat(),
            "today": {
                "requests": 10,
                "tokens": tokens,
                "input_tokens": tokens,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 0,
                "cost": 1.0,
            },
            "providers": [
                {
                    "name": "Codex local - account@example.com",
                    "requests": 10,
                    "tokens": tokens,
                    "input_tokens": tokens,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 1.0,
                    "window_7d": {
                        "requests": 20,
                        "tokens": tokens * 2,
                        "cost": 2.0,
                        "quota_available": True,
                    },
                }
            ],
            "latest_request": {
                "provider": "Codex local - account@example.com",
                "model": "gpt-test",
                "created_at": f"{snapshot_day.isoformat()}T09:00:00+08:00",
                "kind": "success",
            },
            "dashboard": {
                "hourly_today": [
                    {"hour": 9, "requests": 10, "tokens": tokens, "cost": 1.0}
                ]
            },
        }

    def test_same_day_switch_preserves_totals_but_uses_current_quota(self) -> None:
        previous = self.snapshot(self.day, 1_000_000)
        current = self.snapshot(self.day, 100_000)
        previous["providers"][0]["window_7d"].update(
            {"remaining_percent": 31.0, "utilization": 69.0}
        )
        current["providers"][0]["window_7d"].update(
            {"remaining_percent": 17.0, "utilization": 83.0}
        )
        self.output_path.write_text(json.dumps(previous), encoding="utf-8")

        client_usage_export.same_day_output_high_water(current, self.output_path, self.day)

        self.assertEqual(current["today"]["tokens"], 1_000_000)
        self.assertEqual(current["providers"][0]["window_7d"]["tokens"], 200_000)
        self.assertEqual(current["providers"][0]["window_7d"]["utilization"], 83.0)
        self.assertEqual(current["dashboard"]["hourly_today"][0]["tokens"], 1_000_000)
        self.assertEqual(current["latest_request"]["model"], "gpt-test")

    def test_new_day_never_inherits_previous_day_high_water(self) -> None:
        yesterday = self.day - timedelta(days=1)
        previous = self.snapshot(yesterday, 1_000_000)
        current = self.snapshot(self.day, 100_000)
        self.output_path.write_text(json.dumps(previous), encoding="utf-8")

        client_usage_export.same_day_output_high_water(current, self.output_path, self.day)

        self.assertEqual(current["today"]["tokens"], 100_000)
        self.assertEqual(current["providers"][0]["window_7d"]["tokens"], 200_000)

    def test_normal_refresh_preserves_cached_30d_account_window(self) -> None:
        previous = self.snapshot(self.day, 1_000_000)
        current = self.snapshot(self.day, 1_100_000)
        previous["account_30d_updated_at"] = f"{self.day.isoformat()}T09:00:00+08:00"
        previous["providers"][0]["window_30d"] = {
            "requests": 120,
            "tokens": 88_000_000,
            "cost": 84.0,
        }
        self.output_path.write_text(json.dumps(previous), encoding="utf-8")

        client_usage_export.same_day_output_high_water(current, self.output_path, self.day)

        self.assertEqual(
            current["account_30d_updated_at"],
            previous["account_30d_updated_at"],
        )
        self.assertEqual(
            current["providers"][0]["window_30d"]["tokens"],
            88_000_000,
        )


class WindowSemanticsTests(unittest.TestCase):
    def test_30d_window_uses_rolling_account_usage(self) -> None:
        now = datetime(2026, 6, 23, 12, 0, 0)
        label = "Codex local - account@example.com"
        rolling_30d = client_usage_export.UsageBucket(
            requests=120,
            input_tokens=88_000_000,
            cost=84.0,
        )
        rolling_7d = client_usage_export.UsageBucket(
            requests=30,
            input_tokens=20_000_000,
            cost=19.0,
        )

        def scan_accounts(_home: Path, start: datetime, _end: datetime):
            return {label: rolling_30d if now - start > timedelta(days=20) else rolling_7d}

        aligned = ({}, {}, {}, {}, {}, {}, {})
        with (
            patch.object(client_usage_export, "cockpit_codex_quota_by_label", return_value={}),
            patch.object(client_usage_export, "cockpit_codex_speed_by_label", return_value={}),
            patch.object(client_usage_export, "scan_cockpit_codex_accounts", side_effect=scan_accounts),
            patch.object(client_usage_export, "scan_cockpit_codex_quota_windows", return_value=aligned),
            patch.object(client_usage_export, "all_cockpit_codex_account_labels", return_value=[label]),
        ):
            result = client_usage_export.build_codex_window_stats(
                Path("."),
                Path("."),
                now,
                {},
                label,
                include_30d=True,
            )

        window_30d = result[label]["window_30d"]
        self.assertEqual(window_30d["requests"], 120)
        self.assertEqual(window_30d["tokens"], 88_000_000)
        self.assertEqual(window_30d["cost"], 84.0)
        self.assertTrue(window_30d["start_at"].startswith("2026-05-24T12:00:00"))

    def test_full_unused_5h_quota_waits_for_first_request(self) -> None:
        window = {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "quota_available": True,
            "quota_stale": False,
            "remaining_percent": 99.0,
            "utilization": 1.0,
            "resets_at": "2026-06-23T17:00:00+08:00",
        }

        client_usage_export.apply_5h_countdown_state(window)

        self.assertTrue(window["quota_idle"])
        self.assertFalse(window["countdown_active"])

        window["requests"] = 1
        window["tokens"] = 100
        client_usage_export.apply_5h_countdown_state(window)

        self.assertFalse(window["quota_idle"])
        self.assertTrue(window["countdown_active"])

    def test_quota_window_start_allows_small_clock_skew(self) -> None:
        now = datetime(2026, 6, 29, 11, 0, 0)
        window = {
            "quota_available": True,
            "quota_stale": False,
            "resets_at": "2026-06-29T14:22:53+08:00",
        }

        start = client_usage_export.quota_window_start(window, now, timedelta(hours=5))

        self.assertIsNotNone(start)
        assert start is not None
        self.assertTrue(start <= datetime(2026, 6, 29, 9, 22, 51))

    def test_quota_windows_use_quota_cycle_boundaries(self) -> None:
        now = datetime(2026, 6, 22, 12, 0, 0)
        label = "Codex local - account@example.com"
        rolling_5h = client_usage_export.UsageBucket(
            requests=8,
            input_tokens=5_000_000,
            cost=5.0,
        )
        rolling_7d = client_usage_export.UsageBucket(
            requests=70,
            input_tokens=66_000_000,
            cost=62.0,
        )
        quota_cycle_5h = client_usage_export.UsageBucket(
            requests=4,
            input_tokens=2_000_000,
            cost=2.0,
        )
        quota_cycle_7d = client_usage_export.UsageBucket(
            requests=40,
            input_tokens=40_000_000,
            cost=38.0,
        )
        quota = {
            label: {
                "window_5h": {
                    "quota_available": True,
                    "remaining_percent": 10.0,
                    "utilization": 90.0,
                    "resets_at": "2026-06-22T14:00:00+08:00",
                },
                "window_7d": {
                    "quota_available": True,
                    "remaining_percent": 17.0,
                    "utilization": 83.0,
                    "resets_at": "2026-06-25T15:00:00+08:00",
                },
            }
        }

        def scan_accounts(_home: Path, start: datetime, _end: datetime):
            return {label: rolling_7d if now - start > timedelta(days=1) else rolling_5h}

        aligned = (
            {label: quota_cycle_5h},
            {label: quota_cycle_7d},
            {},
            {label: now - timedelta(hours=2)},
            {label: datetime(2026, 6, 18, 15, 0, 0)},
            {},
            {},
        )
        with (
            patch.object(client_usage_export, "cockpit_codex_quota_by_label", return_value=quota),
            patch.object(client_usage_export, "cockpit_codex_speed_by_label", return_value={}),
            patch.object(client_usage_export, "scan_cockpit_codex_accounts", side_effect=scan_accounts),
            patch.object(client_usage_export, "scan_cockpit_codex_quota_windows", return_value=aligned),
            patch.object(client_usage_export, "all_cockpit_codex_account_labels", return_value=[label]),
        ):
            result = client_usage_export.build_codex_window_stats(
                Path("."),
                Path("."),
                now,
                {},
                label,
            )

        window_5h = result[label]["window_5h"]
        window_7d = result[label]["window_7d"]
        self.assertEqual(window_5h["tokens"], 2_000_000)
        self.assertEqual(window_5h["utilization"], 90.0)
        self.assertTrue(window_5h["start_at"].startswith("2026-06-22T10:00:00"))
        self.assertEqual(window_7d["tokens"], 40_000_000)
        self.assertEqual(window_7d["utilization"], 83.0)
        self.assertTrue(window_7d["start_at"].startswith("2026-06-18T15:00:00"))

    def test_7d_without_quota_remains_rolling(self) -> None:
        now = datetime(2026, 6, 22, 12, 0, 0)
        label = "Codex local - account@example.com"
        rolling_7d = client_usage_export.UsageBucket(
            requests=70,
            input_tokens=66_000_000,
            cost=62.0,
        )

        def scan_accounts(_home: Path, start: datetime, _end: datetime):
            return {label: rolling_7d}

        aligned = ({}, {}, {}, {}, {}, {}, {})
        with (
            patch.object(client_usage_export, "cockpit_codex_quota_by_label", return_value={}),
            patch.object(client_usage_export, "cockpit_codex_speed_by_label", return_value={}),
            patch.object(client_usage_export, "scan_cockpit_codex_accounts", side_effect=scan_accounts),
            patch.object(client_usage_export, "scan_cockpit_codex_quota_windows", return_value=aligned),
            patch.object(client_usage_export, "all_cockpit_codex_account_labels", return_value=[label]),
        ):
            result = client_usage_export.build_codex_window_stats(
                Path("."),
                Path("."),
                now,
                {},
                label,
            )

        window_7d = result[label]["window_7d"]
        self.assertEqual(window_7d["tokens"], 66_000_000)
        self.assertTrue(window_7d["start_at"].startswith("2026-06-15T12:00:00"))


class CodexSessionModelTests(unittest.TestCase):
    def write_session(self, root: Path, name: str, rows: list[dict]) -> None:
        session_dir = root / "2026" / "07" / "12"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / name).write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )

    def token_count(self, timestamp: str, input_tokens: int, output_tokens: int) -> dict:
        return {
            "timestamp": timestamp,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": 0,
                        "output_tokens": output_tokens,
                    },
                    "total_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": 0,
                        "output_tokens": output_tokens,
                    },
                },
            },
        }

    def test_token_count_inherits_model_from_turn_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_session(
                root,
                "rollout-session-1.jsonl",
                [
                    {
                        "timestamp": "2026-07-12T09:59:00",
                        "type": "session_meta",
                        "payload": {"id": "session-1"},
                    },
                    {
                        "timestamp": "2026-07-12T10:00:00",
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.6-sol"},
                    },
                    self.token_count("2026-07-12T10:01:00", 100, 10),
                ],
            )

            events = client_usage_export.scan_codex_events(
                root,
                datetime(2026, 7, 12, 9, 0),
                datetime(2026, 7, 12, 11, 0),
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].model, "gpt-5.6-sol")

    def test_model_switch_only_applies_to_later_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.token_count("2026-07-12T10:01:00", 100, 10)
            second = self.token_count("2026-07-12T10:03:00", 200, 20)
            second["payload"]["info"]["total_token_usage"].update(
                {"input_tokens": 300, "output_tokens": 30}
            )
            self.write_session(
                root,
                "rollout-session-2.jsonl",
                [
                    {
                        "timestamp": "2026-07-12T09:59:00",
                        "type": "session_meta",
                        "payload": {"id": "session-2"},
                    },
                    {
                        "timestamp": "2026-07-12T10:00:00",
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.6-sol"},
                    },
                    first,
                    {
                        "timestamp": "2026-07-12T10:02:00",
                        "type": "turn_context",
                        "payload": {"model": "gpt-5.4"},
                    },
                    second,
                ],
            )

            events = client_usage_export.scan_codex_events(
                root,
                datetime(2026, 7, 12, 9, 0),
                datetime(2026, 7, 12, 11, 0),
            )

        self.assertEqual([event.model for event in events], ["gpt-5.6-sol", "gpt-5.4"])

    def test_only_terminal_turn_errors_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_session(
                root,
                "rollout-failures.jsonl",
                [
                    {
                        "timestamp": "2026-07-12T02:50:00",
                        "type": "session_meta",
                        "payload": {"id": "failure-session"},
                    },
                    {
                        "timestamp": "2026-07-12T03:00:00",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "retry-turn"},
                    },
                    {
                        "timestamp": "2026-07-12T03:05:00",
                        "type": "event_msg",
                        "payload": {"type": "stream_error", "message": "retrying"},
                    },
                    {
                        "timestamp": "2026-07-12T03:06:00",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "retry-turn"},
                    },
                    {
                        "timestamp": "2026-07-12T03:10:00",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "interrupted-turn"},
                    },
                    {
                        "timestamp": "2026-07-12T03:11:00",
                        "type": "event_msg",
                        "payload": {
                            "type": "turn_aborted",
                            "turn_id": "interrupted-turn",
                            "reason": "interrupted",
                        },
                    },
                    {
                        "timestamp": "2026-07-12T03:20:00",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "legacy-error"},
                    },
                    {
                        "timestamp": "2026-07-12T03:21:00",
                        "type": "event_msg",
                        "payload": {
                            "type": "error",
                            "message": "stream disconnected",
                            "codex_error_info": {
                                "response_stream_disconnected": {"http_status_code": 502}
                            },
                        },
                    },
                    {
                        "timestamp": "2026-07-12T03:22:00",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "legacy-error"},
                    },
                    {
                        "timestamp": "2026-07-12T03:30:00",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "terminal-error"},
                    },
                    {
                        "timestamp": "2026-07-12T03:31:00",
                        "type": "event_msg",
                        "payload": {
                            "type": "task_complete",
                            "turn_id": "terminal-error",
                            "error": {
                                "message": "server failed",
                                "codex_error_info": "internal_server_error",
                            },
                        },
                    },
                ],
            )
            failures: list[client_usage_export.CodexFailureEvent] = []

            client_usage_export.scan_codex_events(
                root,
                datetime(2026, 7, 12, 2, 0),
                datetime(2026, 7, 12, 4, 0),
                failure_events=failures,
            )

        self.assertEqual([failure.turn_id for failure in failures], ["legacy-error", "terminal-error"])
        self.assertEqual([failure.when.minute for failure in failures], [22, 31])

    def test_error_marks_only_the_adjacent_observed_zero_hour(self) -> None:
        hourly = [
            {"hour": hour, "requests": 0, "tokens": 0, "cost": 0.0}
            for hour in range(24)
        ]
        hourly[3].update({"requests": 137, "tokens": 23_100_000})
        failures = [
            client_usage_export.CodexFailureEvent(
                when=datetime(2026, 7, 12, 3, 55),
                session_id="failure-session",
                turn_id="failed-turn",
            )
        ]

        client_usage_export.mark_codex_failure_hours(
            hourly,
            failures,
            date(2026, 7, 12),
            datetime(2026, 7, 12, 4, 20),
        )

        self.assertTrue(hourly[4]["failure"])
        self.assertEqual(hourly[4]["failure_count"], 1)
        self.assertFalse(any(row.get("failure") for row in hourly[5:]))

    def test_error_marker_waits_for_the_hour_and_clears_when_activity_resumes(self) -> None:
        hourly = [
            {"hour": hour, "requests": 0, "tokens": 0, "cost": 0.0}
            for hour in range(24)
        ]
        hourly[3]["tokens"] = 100
        failures = [
            client_usage_export.CodexFailureEvent(
                when=datetime(2026, 7, 12, 3, 55),
                session_id="failure-session",
                turn_id="failed-turn",
            )
        ]

        client_usage_export.mark_codex_failure_hours(
            hourly,
            failures,
            date(2026, 7, 12),
            datetime(2026, 7, 12, 3, 59),
        )
        self.assertFalse(any(row.get("failure") for row in hourly))

        hourly[4]["tokens"] = 50
        client_usage_export.mark_codex_failure_hours(
            hourly,
            failures,
            date(2026, 7, 12),
            datetime(2026, 7, 12, 5, 30),
        )
        self.assertFalse(any(row.get("failure") for row in hourly))


class AttributionLedgerTests(unittest.TestCase):
    def test_stable_event_id_wins_when_route_time_changes(self) -> None:
        event = client_usage_export.UsageEvent(
            when=datetime(2026, 6, 23, 16, 49, 45),
            request_at=datetime(2026, 6, 23, 16, 2, 46),
            model="gpt-test",
            input_tokens=100,
            cached_tokens=200,
            output_tokens=10,
            session_id="session-1",
        )
        stable_id = client_usage_export.codex_event_id(event)
        legacy_id = client_usage_export.legacy_codex_event_id(event)
        ledger = {
            stable_id: "Codex local - new-account@example.com",
            legacy_id: "Codex local - old-account@example.com",
        }

        attributed = client_usage_export.attribute_codex_events_by_account(
            [event],
            [],
            ledger,
        )

        self.assertIn("Codex local - new-account@example.com", attributed)
        self.assertNotIn("Codex local - old-account@example.com", attributed)

    def test_legacy_event_id_is_migrated_without_losing_attribution(self) -> None:
        event = client_usage_export.UsageEvent(
            when=datetime(2026, 6, 23, 16, 49, 45),
            request_at=datetime(2026, 6, 23, 16, 2, 46),
            model="gpt-test",
            input_tokens=100,
            cached_tokens=200,
            output_tokens=10,
            session_id="session-1",
        )
        stable_id = client_usage_export.codex_event_id(event)
        legacy_id = client_usage_export.legacy_codex_event_id(event)
        ledger = {legacy_id: "Codex local - account@example.com"}

        attributed = client_usage_export.attribute_codex_events_by_account(
            [event],
            [],
            ledger,
        )

        self.assertIn("Codex local - account@example.com", attributed)
        self.assertEqual(ledger[stable_id], "Codex local - account@example.com")


if __name__ == "__main__":
    unittest.main()
