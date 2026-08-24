import json
import hashlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import client_usage_export as usage


def local_event(
    when: datetime,
    *,
    input_tokens: int = 100,
    cached_tokens: int = 200,
    output_tokens: int = 30,
    session_id: str = "session-a",
) -> usage.UsageEvent:
    return usage.UsageEvent(
        when=when,
        model="gpt-5.6-sol",
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        session_id=session_id,
    )


def marker(
    when: datetime,
    label: str,
    *,
    input_tokens: int = 100,
    cached_tokens: int = 200,
    output_tokens: int = 30,
    conversation_id: str = "",
    request_id: str = "",
) -> usage.OpenCodexUsageMarker:
    return usage.OpenCodexUsageMarker(
        request_at=when - timedelta(seconds=1),
        when=when,
        model="gpt-5.6-sol",
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + cached_tokens + output_tokens,
        label=label,
        account_log_label="main",
        conversation_id=conversation_id,
        request_id=request_id,
    )


class OpenCodexAccountAttributionTests(unittest.TestCase):
    def test_compactor_ignores_zero_usage_but_keeps_cancelled_usage(self) -> None:
        zero = usage.compact_opencodex_usage_row(
            {
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "status": 503,
                "usage": {
                    "inputTokens": 0,
                    "cachedInputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 0,
                },
            }
        )
        cancelled = usage.compact_opencodex_usage_row(
            {
                "provider": "openai-pabc",
                "model": "gpt-5.6-sol",
                "status": 499,
                "timestamp": 1_800_000_000_000,
                "durationMs": 1200,
                "accountLogLabel": "pabc",
                "usage": {
                    "inputTokens": 300,
                    "cachedInputTokens": 200,
                    "outputTokens": 30,
                    "totalTokens": 330,
                },
            }
        )

        self.assertIsNone(zero)
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled["input_tokens"], 100)
        self.assertEqual(cancelled["cached_tokens"], 200)
        self.assertEqual(cancelled["total_tokens"], 330)

    def test_compactor_uses_final_positive_attempt_account(self) -> None:
        compact = usage.compact_opencodex_usage_row(
            {
                "provider": "openai-pa00000",
                "model": "gpt-5.6-sol",
                "accountLogLabel": "pa00000",
                "timestamp": 1_800_000_000_000,
                "durationMs": 1200,
                "requestId": "request-a",
                "usage": {
                    "inputTokens": 330,
                    "cachedInputTokens": 200,
                    "outputTokens": 40,
                    "totalTokens": 370,
                },
                "attempts": [
                    {
                        "ordinal": 1,
                        "provider": "openai-pa00000",
                        "model": "gpt-5.6-sol",
                        "accountLogLabel": "pa00000",
                        "status": 429,
                        "usage": {
                            "inputTokens": 0,
                            "cachedInputTokens": 0,
                            "outputTokens": 0,
                            "totalTokens": 0,
                        },
                    },
                    {
                        "ordinal": 2,
                        "provider": "openai-pb00000",
                        "model": "gpt-5.6-sol",
                        "accountLogLabel": "pb00000",
                        "status": 200,
                        "usage": {
                            "inputTokens": 300,
                            "cachedInputTokens": 200,
                            "outputTokens": 30,
                            "totalTokens": 330,
                        },
                    },
                ],
            }
        )

        self.assertIsNotNone(compact)
        self.assertEqual(compact["account_log_label"], "pb00000")
        self.assertEqual(compact["attempt_ordinal"], 2)
        self.assertEqual(compact["total_tokens"], 330)

    def test_named_log_label_uses_config_and_rejects_reused_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            root = home / ".opencodex"
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "codexAccounts": [
                            {
                                "logLabel": "pstable",
                                "email": "stable@example.com",
                                "planType": "plus",
                            },
                            {
                                "logLabel": "preused",
                                "email": "new@example.com",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            backup = root / "config.json.old"
            backup.write_text(
                json.dumps(
                    {
                        "codexAccounts": [
                            {
                                "logLabel": "preused",
                                "email": "old@example.com",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            mapping = usage.opencodex_log_label_accounts(home)

        self.assertEqual(
            mapping["pstable"],
            ("Codex local - stable@example.com", "plus"),
        )
        self.assertNotIn("preused", mapping)

    def test_main_label_only_applies_after_confirmed_switch_edge(self) -> None:
        edge = datetime(2026, 8, 23, 10, 0, 0)
        snapshots = [
            usage.OpenCodexAccountSnapshot(
                when=edge,
                label="Codex local - new@example.com",
                plan_type="plus",
            )
        ]
        rows = []
        for request_at in (edge - timedelta(seconds=1), edge + timedelta(seconds=1)):
            rows.append(
                {
                    "timestamp": usage.local_epoch_ms(request_at),
                    "duration_ms": 1000,
                    "model": "gpt-5.6-sol",
                    "account_log_label": "main",
                    "input_tokens": 100,
                    "cached_tokens": 200,
                    "output_tokens": 30,
                    "total_tokens": 330,
                }
            )

        markers = usage.scan_opencodex_usage_markers(
            Path("unused"),
            edge - timedelta(minutes=1),
            edge + timedelta(minutes=1),
            rows=rows,
            snapshots=snapshots,
            log_label_accounts={},
        )

        self.assertEqual(markers[0].label, "")
        self.assertEqual(markers[1].label, "Codex local - new@example.com")

    def test_main_route_ignores_disagreeing_opencodex_auth_account(self) -> None:
        request_at = datetime(2026, 8, 23, 10, 30, 0)
        switch_at = request_at - timedelta(minutes=30)
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            codex_root = home / ".codex"
            codex_root.mkdir()
            (codex_root / "config.toml").write_text(
                'model_provider = "codex_local_access"\n',
                encoding="utf-8",
            )
            codex_auth = codex_root / "auth.json"
            codex_auth.write_text(
                json.dumps({"email": "will@example.com"}),
                encoding="utf-8",
            )
            os.utime(codex_auth, (switch_at.timestamp(), switch_at.timestamp()))
            (codex_root / ".cockpit_codex_auth.json").write_text(
                json.dumps({"email": "wrong-cockpit@example.com"}),
                encoding="utf-8",
            )

            opencodex_root = home / ".opencodex"
            opencodex_root.mkdir()
            (opencodex_root / "auth.json").write_text(
                json.dumps(
                    {
                        "chatgpt": {
                            "activeAccountId": "wrong",
                            "accounts": [
                                {
                                    "id": "wrong",
                                    "credential": {"email": "wrong@example.com"},
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            account_timeline = home / "account-timeline.json"
            account_timeline.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "at": switch_at.isoformat(),
                                "label": "Codex local - will@example.com",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stale_opencodex_timeline = home / "opencodex-timeline.json"
            stale_opencodex_timeline.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "at": switch_at.isoformat(),
                                "label": "Codex local - wrong@example.com",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "timestamp": usage.local_epoch_ms(request_at),
                "duration_ms": 1000,
                "model": "gpt-5.6-sol",
                "account_log_label": "main",
                "input_tokens": 100,
                "cached_tokens": 200,
                "output_tokens": 30,
                "total_tokens": 330,
            }
            with (
                mock.patch.object(usage, "ACCOUNT_TIMELINE_PATH", account_timeline),
                mock.patch.object(
                    usage,
                    "AUTH_SWITCH_EVENTS_PATH",
                    home / "auth-switches.jsonl",
                ),
                mock.patch.object(
                    usage,
                    "OPENCODEX_ACCOUNT_TIMELINE_PATH",
                    stale_opencodex_timeline,
                ),
            ):
                markers = usage.scan_opencodex_usage_markers(
                    home,
                    request_at - timedelta(seconds=1),
                    request_at + timedelta(seconds=2),
                    rows=[row],
                    log_label_accounts={},
                )

        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].label, "Codex local - will@example.com")

    def test_exact_usage_hint_overrides_stale_ledger_without_changing_tokens(self) -> None:
        when = datetime(2026, 8, 23, 11, 0, 0)
        event = local_event(when)
        event_id = usage.codex_event_id(event)
        ledger = {event_id: "Codex local - stale@example.com"}
        before = event.total_tokens

        matched = usage.apply_opencodex_account_hints(
            Path("unused"),
            [event],
            when - timedelta(minutes=1),
            when + timedelta(minutes=1),
            markers=[marker(when, "Codex local - final@example.com")],
        )
        attributed = usage.attribute_codex_events_by_account(
            [event],
            [],
            ledger,
        )

        self.assertEqual(matched, 1)
        self.assertEqual(event.total_tokens, before)
        self.assertEqual(list(attributed), ["Codex local - final@example.com"])
        self.assertEqual(ledger[event_id], "Codex local - final@example.com")

    def test_near_concurrent_equal_token_events_map_one_to_one(self) -> None:
        base = datetime(2026, 8, 23, 12, 0, 0)
        first = local_event(base, session_id="first")
        second = local_event(base + timedelta(milliseconds=220), session_id="second")
        markers = [
            marker(base + timedelta(milliseconds=5), "Codex local - a@example.com"),
            marker(base + timedelta(milliseconds=225), "Codex local - b@example.com"),
        ]

        usage.apply_opencodex_account_hints(
            Path("unused"),
            [first, second],
            base - timedelta(seconds=1),
            base + timedelta(seconds=1),
            markers=markers,
        )

        self.assertEqual(first.account_label_hint, "Codex local - a@example.com")
        self.assertEqual(second.account_label_hint, "Codex local - b@example.com")

    def test_conversation_id_prevents_cross_session_equal_token_swap(self) -> None:
        base = datetime(2026, 8, 23, 12, 30, 0)
        first = local_event(base, session_id="first-session")
        second = local_event(base + timedelta(milliseconds=100), session_id="second-session")
        first_conversation = hashlib.sha256(b"first-session").hexdigest()[:32]
        second_conversation = hashlib.sha256(b"second-session").hexdigest()[:32]
        markers = [
            marker(
                second.when,
                "Codex local - a@example.com",
                conversation_id=first_conversation,
            ),
            marker(
                first.when,
                "Codex local - b@example.com",
                conversation_id=second_conversation,
            ),
        ]

        usage.apply_opencodex_account_hints(
            Path("unused"),
            [first, second],
            base - timedelta(seconds=1),
            base + timedelta(seconds=1),
            markers=markers,
        )

        self.assertEqual(first.account_label_hint, "Codex local - a@example.com")
        self.assertEqual(second.account_label_hint, "Codex local - b@example.com")

    def test_child_session_uses_root_thread_for_opencodex_conversation_id(self) -> None:
        root_thread_id = "019eca6f-9d4e-7992-bbd9-17921f96202c"
        rows = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "01a02dbe-child-session",
                    "session_id": root_thread_id,
                    "parent_thread_id": root_thread_id,
                },
            }
        ]

        conversation_id = usage.codex_opencodex_conversation_id_from_rows(rows)

        self.assertEqual(
            conversation_id,
            hashlib.sha256(root_thread_id.encode("utf-8")).hexdigest()[:32],
        )

    def test_confirmed_account_propagates_only_inside_same_turn(self) -> None:
        base = datetime(2026, 8, 23, 12, 45, 0)
        turn_start = base - timedelta(seconds=5)
        anchor = local_event(base, session_id="same-session")
        anchor.account_at = turn_start
        sibling = local_event(
            base + timedelta(seconds=1),
            input_tokens=101,
            session_id="same-session",
        )
        sibling.account_at = turn_start
        next_turn = local_event(
            base + timedelta(seconds=2),
            input_tokens=102,
            session_id="same-session",
        )
        next_turn.account_at = turn_start + timedelta(seconds=10)

        usage.apply_opencodex_account_hints(
            Path("unused"),
            [anchor, sibling, next_turn],
            base - timedelta(seconds=10),
            base + timedelta(seconds=10),
            markers=[marker(base, "Codex local - a@example.com")],
        )

        self.assertEqual(anchor.account_hint_source, usage.OPENCODEX_ACCOUNT_HINT_SOURCE)
        self.assertEqual(sibling.account_hint_source, usage.OPENCODEX_TURN_HINT_SOURCE)
        self.assertEqual(sibling.account_label_hint, "Codex local - a@example.com")
        self.assertEqual(next_turn.account_hint_source, "")

    def test_same_turn_account_reselection_uses_latest_confirmed_account(self) -> None:
        base = datetime(2026, 8, 23, 12, 50, 0)
        turn_start = base - timedelta(seconds=5)
        first_account = local_event(base, session_id="same-session")
        first_account.account_at = turn_start
        second_account = local_event(
            base + timedelta(seconds=2),
            input_tokens=101,
            session_id="same-session",
        )
        second_account.account_at = turn_start
        after_reselection = local_event(
            base + timedelta(seconds=3),
            input_tokens=102,
            session_id="same-session",
        )
        after_reselection.account_at = turn_start

        usage.apply_opencodex_account_hints(
            Path("unused"),
            [first_account, second_account, after_reselection],
            base - timedelta(seconds=10),
            base + timedelta(seconds=10),
            markers=[
                marker(base, "Codex local - a@example.com"),
                marker(
                    base + timedelta(seconds=2),
                    "Codex local - b@example.com",
                    input_tokens=101,
                ),
            ],
        )

        self.assertEqual(first_account.account_label_hint, "Codex local - a@example.com")
        self.assertEqual(second_account.account_label_hint, "Codex local - b@example.com")
        self.assertEqual(
            after_reselection.account_hint_source,
            usage.OPENCODEX_TURN_HINT_SOURCE,
        )
        self.assertEqual(
            after_reselection.account_label_hint,
            "Codex local - b@example.com",
        )

    def test_equal_distance_different_accounts_remains_unmatched(self) -> None:
        when = datetime(2026, 8, 23, 13, 0, 0)
        event = local_event(when)
        markers = [
            marker(when - timedelta(seconds=1), "Codex local - a@example.com"),
            marker(when + timedelta(seconds=1), "Codex local - b@example.com"),
        ]

        matched = usage.apply_opencodex_account_hints(
            Path("unused"),
            [event],
            when - timedelta(seconds=2),
            when + timedelta(seconds=2),
            markers=markers,
        )

        self.assertEqual(matched, 0)
        self.assertEqual(event.account_hint_source, "")

    def test_unresolved_opencodex_row_does_not_fall_back_to_official_account(self) -> None:
        when = datetime(2026, 8, 23, 14, 0, 0)
        event = local_event(when)
        unresolved = marker(when, "")
        usage.apply_opencodex_account_hints(
            Path("unused"),
            [event],
            when - timedelta(seconds=1),
            when + timedelta(seconds=1),
            markers=[unresolved],
        )

        attributed = usage.attribute_codex_events_by_account(
            [event],
            [],
            {},
            current_label="Codex local - wrong@example.com",
            now=when,
        )

        self.assertEqual(list(attributed), [usage.API_SERVICE_AGGREGATE_LABEL])

    def test_deleted_named_account_keeps_anonymous_stable_label(self) -> None:
        edge = datetime(2026, 8, 23, 14, 30, 0)
        rows = [
            {
                "timestamp": usage.local_epoch_ms(edge),
                "duration_ms": 1000,
                "model": "gpt-5.6-sol",
                "account_log_label": "pabc123",
                "request_id": "request-a",
                "input_tokens": 100,
                "cached_tokens": 200,
                "output_tokens": 30,
                "total_tokens": 330,
            }
        ]

        markers = usage.scan_opencodex_usage_markers(
            Path("unused"),
            edge - timedelta(seconds=1),
            edge + timedelta(seconds=2),
            rows=rows,
            snapshots=[],
            log_label_accounts={},
        )

        self.assertEqual(markers[0].label, "Codex local - OpenCodex-pabc123")

    def test_confirmed_opencodex_route_without_exact_usage_stays_unresolved(self) -> None:
        when = datetime(2026, 8, 23, 14, 45, 0)
        event = local_event(when, input_tokens=999, session_id="route-session")
        conversation_id = hashlib.sha256(b"route-session").hexdigest()[:32]
        route_marker = marker(
            when + timedelta(milliseconds=100),
            "Codex local - a@example.com",
            conversation_id=conversation_id,
        )

        matched = usage.apply_opencodex_account_hints(
            Path("unused"),
            [event],
            when - timedelta(seconds=1),
            when + timedelta(seconds=1),
            markers=[route_marker],
        )

        self.assertEqual(matched, 0)
        self.assertEqual(
            event.account_hint_source,
            usage.OPENCODEX_UNRESOLVED_HINT_SOURCE,
        )
        self.assertEqual(event.account_label_hint, usage.API_SERVICE_AGGREGATE_LABEL)

    def test_opencodex_hint_beats_stale_cockpit_usage_row(self) -> None:
        when = datetime(2026, 8, 23, 15, 0, 0)
        event = local_event(when)
        event.account_label_hint = "Codex local - final@example.com"
        event.account_hint_source = usage.OPENCODEX_ACCOUNT_HINT_SOURCE
        cockpit_marker = usage.AccountMarker(
            when=when,
            label="Codex local - stale@example.com",
            model=event.model,
            total_tokens=event.total_tokens,
            input_tokens=event.input_tokens + event.cached_tokens,
            cached_tokens=event.cached_tokens,
            output_tokens=event.output_tokens,
        )
        verdicts = {}

        resolved, sessions, unresolved = usage.resolve_api_service_event_accounts(
            {"Codex local - stale@example.com": [event]},
            [cockpit_marker],
            verdicts=verdicts,
        )

        self.assertEqual(list(resolved), ["Codex local - final@example.com"])
        self.assertEqual(sessions["session-a"], "Codex local - final@example.com")
        self.assertEqual(unresolved, 0)
        self.assertEqual(
            verdicts[usage.codex_event_id(event)]["tier"],
            "opencodex_usage_row",
        )

    def test_main_account_timeline_uses_codex_auth_and_ignores_opencodex_auth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            codex_root = home / ".codex"
            codex_root.mkdir()
            (codex_root / "config.toml").write_text(
                'model_provider = "openai"\n',
                encoding="utf-8",
            )
            opencodex_root = home / ".opencodex"
            opencodex_root.mkdir()
            (opencodex_root / "auth.json").write_text(
                json.dumps(
                    {
                        "chatgpt": {
                            "activeAccountId": "wrong",
                            "accounts": [
                                {
                                    "id": "wrong",
                                    "credential": {
                                        "email": "wrong-opencodex@example.com"
                                    },
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            auth_path = codex_root / "auth.json"
            timeline_path = home / "timeline.json"
            auth_events_path = home / "auth-switches.jsonl"

            def write_auth(account_id: str, email: str, changed_at: datetime) -> None:
                auth_path.write_text(
                    json.dumps({"email": email, "account_id": account_id}),
                    encoding="utf-8",
                )
                stamp = changed_at.timestamp()
                os.utime(auth_path, (stamp, stamp))

            first_at = datetime(2026, 8, 23, 9, 0, 0)
            second_at = datetime(2026, 8, 23, 10, 0, 0)
            with (
                mock.patch.object(usage, "ACCOUNT_TIMELINE_PATH", timeline_path),
                mock.patch.object(
                    usage,
                    "AUTH_SWITCH_EVENTS_PATH",
                    auth_events_path,
                ),
            ):
                write_auth("a", "a@example.com", first_at)
                usage.record_current_opencodex_account_snapshot(
                    home,
                    first_at + timedelta(minutes=1),
                )
                write_auth("a", "a@example.com", first_at + timedelta(minutes=5))
                usage.record_current_opencodex_account_snapshot(
                    home,
                    first_at + timedelta(minutes=6),
                )
                write_auth("b", "b@example.com", second_at)
                usage.record_current_opencodex_account_snapshot(
                    home,
                    second_at + timedelta(minutes=1),
                )
                snapshots = usage.opencodex_account_snapshots(
                    home,
                    second_at + timedelta(minutes=1),
                )

        self.assertEqual(
            [(item.when, item.label) for item in snapshots],
            [
                (first_at, "Codex local - a@example.com"),
                (second_at, "Codex local - b@example.com"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
