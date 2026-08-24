import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import client_usage_export as usage


BASE = datetime(2026, 8, 23, 12, 0, tzinfo=usage.LOCAL_TZ)


def local_event(
    when: datetime = BASE,
    *,
    model: str = "gpt-5.6-sol",
    input_tokens: int = 100,
    cached_tokens: int = 200,
    output_tokens: int = 30,
    conversation_id: str = "conversation-a",
    session_id: str = "session-a",
    request_key: str = "",
    route: str = "opencodex",
) -> usage.UsageEvent:
    return usage.UsageEvent(
        when=when,
        model=model,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        conversation_id=conversation_id,
        session_id=session_id,
        request_key=request_key,
        route=route,
    )


def proxy_marker(
    when: datetime = BASE,
    *,
    model: str = "gpt-5.6-sol",
    input_tokens: int = 100,
    cached_tokens: int = 200,
    output_tokens: int = 30,
    conversation_id: str = "conversation-a",
    request_id: str = "request-a",
    attempt_ordinal: int = 1,
    label: str = "Codex local - account@example.com",
    route_kind: str = "responses",
    inbound_protocol: str = "responses",
    admission_kind: str = "",
) -> usage.OpenCodexUsageMarker:
    return usage.OpenCodexUsageMarker(
        request_at=when - timedelta(seconds=1),
        when=when,
        model=model,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + cached_tokens + output_tokens,
        label=label,
        account_log_label="main" if label else "",
        request_id=request_id,
        conversation_id=conversation_id,
        attempt_ordinal=attempt_ordinal,
        provider="openai-main",
        resolved_model=model,
        route_kind=route_kind,
        inbound_protocol=inbound_protocol,
        admission_kind=admission_kind,
        source_instance="test",
    )


def reconcile(
    local_events: list[usage.UsageEvent],
    markers: list[usage.OpenCodexUsageMarker],
    *,
    start: datetime = BASE - timedelta(days=1),
    end: datetime = BASE + timedelta(days=1),
) -> usage.OpenCodexReconciliationResult:
    return usage.reconcile_opencodex_usage_events(
        local_events,
        markers,
        start,
        end,
    )


class HybridUsageReconciliationTests(unittest.TestCase):
    def test_exact_match_counts_one_physical_request(self) -> None:
        event = local_event(request_key="request-a")
        result = reconcile([event], [proxy_marker()])

        self.assertEqual(len(result.events), 1)
        self.assertEqual(sum(item.total_tokens for item in result.events), 330)
        self.assertEqual(result.events[0].session_id, "session-a")
        self.assertTrue(result.events[0].request_key.startswith("opencodex:test:request-a:"))
        self.assertEqual(result.events[0].reconciliation_status, "matched")
        self.assertEqual(result.diagnostics.requests["matched"], 1)

    def test_exact_request_id_does_not_absorb_official_local_event(self) -> None:
        event = local_event(request_key="request-a", route="official")

        result = reconcile([event], [proxy_marker()])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(
            sorted(item.reconciliation_status for item in result.events),
            ["local_only", "proxy_only"],
        )

    def test_exact_request_id_rejects_marker_five_hours_in_future(self) -> None:
        event = local_event(BASE, request_key="request-a")
        marker = proxy_marker(BASE + timedelta(hours=5))

        result = reconcile([event], [marker])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(
            sorted(item.reconciliation_status for item in result.events),
            ["local_only", "proxy_only"],
        )

    def test_same_conversation_accepts_25_minute_and_5_hour_late_local_events(self) -> None:
        for delay in (timedelta(minutes=25), timedelta(hours=5)):
            with self.subTest(delay=delay):
                marker = proxy_marker()
                event = local_event(BASE + delay)

                result = reconcile([event], [marker])

                self.assertEqual(len(result.events), 1)
                self.assertEqual(result.events[0].reconciliation_status, "matched")

    def test_future_marker_only_allows_60_second_clock_skew(self) -> None:
        accepted = reconcile(
            [local_event(BASE)],
            [proxy_marker(BASE + timedelta(seconds=60))],
        )
        rejected = reconcile(
            [local_event(BASE)],
            [proxy_marker(BASE + timedelta(hours=5))],
        )

        self.assertEqual(len(accepted.events), 1)
        self.assertEqual(accepted.events[0].reconciliation_status, "matched")
        self.assertEqual(len(rejected.events), 2)
        self.assertEqual(
            sorted(item.reconciliation_status for item in rejected.events),
            ["local_only", "proxy_only"],
        )

    def test_same_conversation_rejects_more_than_6_hours(self) -> None:
        result = reconcile(
            [local_event(BASE + timedelta(hours=6, seconds=1))],
            [proxy_marker()],
        )

        self.assertEqual(len(result.events), 2)
        self.assertEqual(
            sorted(item.reconciliation_status for item in result.events),
            ["local_only", "proxy_only"],
        )

    def test_concurrent_identical_signatures_stay_with_their_conversations(self) -> None:
        local_a = local_event(conversation_id="conversation-a", session_id="session-a")
        local_b = local_event(conversation_id="conversation-b", session_id="session-b")
        marker_a = proxy_marker(conversation_id="conversation-a", request_id="request-a")
        marker_b = proxy_marker(conversation_id="conversation-b", request_id="request-b")

        result = reconcile([local_b, local_a], [marker_a, marker_b])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.diagnostics.requests["matched"], 2)
        self.assertEqual(
            {item.conversation_id: item.session_id for item in result.events},
            {"conversation-a": "session-a", "conversation-b": "session-b"},
        )

    def test_missing_conversation_matches_only_bidirectionally_unique_near_pair(self) -> None:
        event = local_event(
            BASE + timedelta(seconds=29),
            conversation_id="",
            session_id="",
        )
        marker = proxy_marker(conversation_id="")

        result = reconcile([event], [marker])

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].reconciliation_status, "matched")

    def test_missing_conversation_ambiguous_candidates_are_not_collapsed(self) -> None:
        event = local_event(conversation_id="", session_id="")
        before = proxy_marker(
            BASE - timedelta(seconds=1),
            conversation_id="",
            request_id="request-before",
        )
        after = proxy_marker(
            BASE + timedelta(seconds=1),
            conversation_id="",
            request_id="request-after",
        )

        result = reconcile([event], [before, after])

        self.assertEqual(len(result.events), 3)
        self.assertEqual(result.diagnostics.requests["matched"], 0)
        self.assertEqual(result.diagnostics.requests["local_only"], 1)
        self.assertEqual(result.diagnostics.requests["proxy_only"], 2)

    def test_same_conversation_ambiguity_preserves_both_physical_requests(self) -> None:
        event = local_event()
        before = proxy_marker(BASE - timedelta(seconds=1), request_id="request-before")
        after = proxy_marker(BASE + timedelta(seconds=1), request_id="request-after")

        result = reconcile([event], [before, after])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.diagnostics.requests["ambiguous"], 1)
        self.assertEqual(result.diagnostics.requests["proxy_only"], 1)
        self.assertEqual(sum(item.total_tokens for item in result.events), 660)

    def test_proxy_only_luna_and_terra_are_admitted(self) -> None:
        luna = proxy_marker(model="gpt-5.6-luna", request_id="luna")
        terra = proxy_marker(
            BASE + timedelta(seconds=1),
            model="gpt-5.6-terra",
            conversation_id="conversation-b",
            request_id="terra",
        )

        result = reconcile([], [luna, terra])

        self.assertEqual(result.diagnostics.requests["proxy_only"], 2)
        self.assertEqual(
            {item.model for item in result.events},
            {"gpt-5.6-luna", "gpt-5.6-terra"},
        )

    def test_empty_conversation_chat_probe_is_not_proxy_usage(self) -> None:
        probe = proxy_marker(
            model="gpt-5.6-terra",
            conversation_id="",
            request_id="model-probe",
            route_kind="chat",
            inbound_protocol="chat",
            admission_kind="model_probe",
        )

        result = reconcile([], [probe])

        self.assertEqual(result.events, [])
        self.assertEqual(result.diagnostics.requests["proxy_only"], 0)

    def test_responses_luna_and_terra_with_conversation_are_real_usage(self) -> None:
        luna = proxy_marker(
            model="gpt-5.6-luna",
            request_id="luna-response",
            inbound_protocol="responses",
        )
        terra = proxy_marker(
            BASE + timedelta(seconds=1),
            model="gpt-5.6-terra",
            conversation_id="conversation-b",
            request_id="terra-response",
            inbound_protocol="responses",
        )

        result = reconcile([], [luna, terra])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.diagnostics.requests["proxy_only"], 2)

    def test_compact_without_local_token_count_is_admitted_from_proxy(self) -> None:
        compact = proxy_marker(
            model="gpt-5.6-luna",
            request_id="compact-a",
            route_kind="responses/compact",
        )

        result = reconcile([], [compact])

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].total_tokens, 330)
        self.assertEqual(result.events[0].usage_provenance, usage.OPENCODEX_USAGE_PROVENANCE)
        self.assertEqual(result.events[0].reconciliation_status, "proxy_only")

    def test_local_only_official_direct_request_is_preserved(self) -> None:
        direct = local_event(route="official")

        result = reconcile([direct], [])

        self.assertEqual(result.events, [direct])
        self.assertEqual(result.events[0].reconciliation_status, "local_only")
        self.assertEqual(result.diagnostics.requests["local_only"], 1)

    def test_official_local_and_same_conversation_proxy_are_two_requests(self) -> None:
        direct = local_event(route="official")
        marker = proxy_marker()

        result = reconcile([direct], [marker])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(
            sorted(item.reconciliation_status for item in result.events),
            ["local_only", "proxy_only"],
        )

    def test_duplicate_log_rows_share_one_canonical_request_key(self) -> None:
        first = proxy_marker()
        duplicate = proxy_marker(BASE + timedelta(milliseconds=1))

        result = reconcile([], [first, duplicate])

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.diagnostics.requests["proxy_only"], 1)

    def test_attempt_ordinal_distinguishes_two_positive_attempts(self) -> None:
        first = proxy_marker(request_id="request-a", attempt_ordinal=1)
        second = proxy_marker(
            BASE + timedelta(seconds=1),
            request_id="request-a",
            attempt_ordinal=2,
        )

        result = reconcile([], [first, second])

        self.assertEqual(len(result.events), 2)
        self.assertEqual(len({item.request_key for item in result.events}), 2)

    def test_zero_usage_attempt_is_dropped_but_cancelled_usage_is_kept(self) -> None:
        zero = usage.compact_opencodex_usage_row(
            {
                "provider": "openai-main",
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
                "provider": "openai-main",
                "model": "gpt-5.6-sol",
                "status": 499,
                "timestamp": 1_800_000_000_000,
                "durationMs": 1_200,
                "requestId": "cancelled-a",
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
        self.assertEqual(cancelled["total_tokens"], 330)

    def test_final_positive_attempt_wins_over_zero_usage_retry(self) -> None:
        compact = usage.compact_opencodex_usage_row(
            {
                "provider": "openai-old",
                "model": "gpt-5.6-sol",
                "timestamp": 1_800_000_000_000,
                "durationMs": 1_200,
                "requestId": "request-a",
                "usage": {
                    "inputTokens": 300,
                    "cachedInputTokens": 200,
                    "outputTokens": 30,
                    "totalTokens": 330,
                },
                "attempts": [
                    {
                        "ordinal": 1,
                        "provider": "openai-old",
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
                        "provider": "openai-final",
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
        self.assertEqual(compact["attempt_ordinal"], 2)
        self.assertEqual(compact["provider"], "openai-final")

    def test_cross_midnight_uses_proxy_completion_date(self) -> None:
        marker_when = datetime(2026, 8, 22, 23, 59, tzinfo=usage.LOCAL_TZ)
        local_when = datetime(2026, 8, 23, 0, 2, tzinfo=usage.LOCAL_TZ)
        start = datetime(2026, 8, 22, 0, 0, tzinfo=usage.LOCAL_TZ)
        end = datetime(2026, 8, 23, 0, 0, tzinfo=usage.LOCAL_TZ)

        result = reconcile(
            [local_event(local_when)],
            [proxy_marker(marker_when)],
            start=start,
            end=end,
        )

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].when, marker_when)
        self.assertEqual(result.events[0].reconciliation_status, "matched")

    def test_without_opencodex_markers_degrades_to_local_ledger(self) -> None:
        first = local_event()
        second = local_event(
            BASE + timedelta(seconds=1),
            conversation_id="conversation-b",
            session_id="session-b",
        )

        result = reconcile([first, second], [])

        self.assertEqual(result.events, [first, second])
        self.assertEqual(result.diagnostics.requests["local_only"], 2)
        self.assertEqual(result.diagnostics.tokens["local_only"], 660)

    def test_unconfirmed_proxy_account_is_not_guessed(self) -> None:
        marker = proxy_marker(label="")

        result = reconcile([], [marker])

        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.account_label_hint, usage.API_SERVICE_AGGREGATE_LABEL)
        self.assertEqual(event.account_hint_source, usage.OPENCODEX_UNRESOLVED_HINT_SOURCE)

    def test_same_second_same_tokens_in_different_sessions_are_not_deduped(self) -> None:
        first = local_event(
            conversation_id="conversation-a",
            session_id="session-a",
            request_key="request-a",
        )
        second = local_event(
            conversation_id="conversation-b",
            session_id="session-b",
            request_key="request-b",
        )

        deduped = usage.dedupe_usage_events([first, second])

        self.assertEqual(len(deduped), 2)
        self.assertEqual({event.session_id for event in deduped}, {"session-a", "session-b"})


class OfflineAccountingMigrationTests(unittest.TestCase):
    @staticmethod
    def history_row(tokens: int, accounting_schema: int = 0) -> dict:
        return {
            "date": "2026-08-22",
            "source": "local",
            "usage_accounting_schema": accounting_schema,
            "cockpit_usage_schema": usage.COCKPIT_USAGE_DEDUPE_SCHEMA,
            "requests": max(1, tokens // 10),
            "tokens": tokens,
            "input_tokens": tokens,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
            "cost": tokens / 100,
            "models": {"gpt-test": tokens},
            "providers": [
                {
                    "name": "Codex local - account@example.com",
                    "tokens": tokens,
                    "input_tokens": tokens,
                    "models": {"gpt-test": tokens},
                }
            ],
            "detail_tokens": tokens,
            "source_gap": {"tokens": 1},
        }

    def test_accounting_migration_targets_only_evidence_dates(self) -> None:
        now = datetime(2026, 8, 24, 9)
        days = {}
        for offset in range(1, 32):
            target = now.date() - timedelta(days=offset)
            row = self.history_row(100)
            row["date"] = target.isoformat()
            days[target.isoformat()] = row
        history = {
            "days": days,
            "offline_sync": {"last_successful_at": now.isoformat()},
        }
        evidence = {now.date() - timedelta(days=offset) for offset in range(1, 9)}

        selected = usage.offline_history_dates_to_reconcile(
            history,
            now,
            max_days=31,
            accounting_evidence_dates=evidence,
        )
        none_selected = usage.offline_history_dates_to_reconcile(
            history,
            now,
            max_days=31,
            accounting_evidence_dates=set(),
        )

        self.assertEqual(selected, sorted(evidence))
        self.assertEqual(none_selected, [])

    def test_accounting_migration_guard_keeps_canonical_details_plus_residual(self) -> None:
        existing = self.history_row(1_000)
        rebuilt = self.history_row(400, usage.USAGE_ACCOUNTING_SCHEMA)

        merged, changed = usage.merge_rebuilt_history_day(existing, rebuilt, "now")

        self.assertTrue(changed)
        self.assertEqual(merged["tokens"], 1_000)
        self.assertEqual(merged["usage_accounting_schema"], 1)
        self.assertNotIn("source_gap", merged)
        self.assertEqual(merged["models"]["gpt-test"], 400)
        self.assertEqual(merged["models"]["Historical high-water"], 600)
        self.assertEqual(sum(merged["models"].values()), 1_000)
        self.assertEqual(
            sum(int(provider.get("tokens") or 0) for provider in merged["providers"]),
            1_000,
        )
        self.assertEqual(
            sum(int(provider.get("requests") or 0) for provider in merged["providers"]),
            merged["requests"],
        )
        self.assertAlmostEqual(
            sum(float(provider.get("cost") or 0) for provider in merged["providers"]),
            merged["cost"],
        )
        self.assertEqual(merged["detail_tokens"], 1_000)

    def test_fresh_queue_status_does_not_spawn_again(self) -> None:
        now = datetime(2026, 8, 24, 9)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            usage,
            "OFFLINE_BACKFILL_STATUS_PATH",
            Path(directory) / "status.json",
        ), patch.object(
            usage,
            "OFFLINE_BACKFILL_LOCK_PATH",
            Path(directory) / "worker.lock",
        ), patch.object(usage, "spawn_offline_backfill_worker", return_value=True) as spawn:
            first = usage.queue_offline_backfill_worker(Path(directory) / "today.json", now)
            second = usage.queue_offline_backfill_worker(Path(directory) / "today.json", now)

        self.assertEqual(first["state"], "queued")
        self.assertEqual(second["run_id"], first["run_id"])
        spawn.assert_called_once()

    def test_idle_status_delays_the_next_worker_check(self) -> None:
        now = datetime(2026, 8, 24, 9)
        status = {
            "state": "idle",
            "next_check_at": "2026-08-24T09:05:00+08:00",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            usage,
            "OFFLINE_BACKFILL_STATUS_PATH",
            Path(directory) / "status.json",
        ), patch.object(
            usage,
            "OFFLINE_BACKFILL_LOCK_PATH",
            Path(directory) / "worker.lock",
        ), patch.object(usage, "spawn_offline_backfill_worker", return_value=True) as spawn:
            usage.write_json_atomic(usage.OFFLINE_BACKFILL_STATUS_PATH, status)
            result = usage.queue_offline_backfill_worker(Path(directory) / "today.json", now)

        self.assertEqual(result, status)
        spawn.assert_not_called()

    def test_future_worker_heartbeat_is_still_a_fresh_lease(self) -> None:
        now = datetime(2026, 8, 24, 9)
        status = {
            "state": "running",
            "run_id": "run-1",
            "heartbeat_at": "2026-08-24T09:00:02+08:00",
        }
        self.assertTrue(usage.offline_backfill_status_is_fresh(status, now))

    def test_launch_failure_does_not_overwrite_a_newer_run(self) -> None:
        now = datetime(2026, 8, 24, 9)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            usage,
            "OFFLINE_BACKFILL_STATUS_PATH",
            Path(directory) / "status.json",
        ), patch.object(
            usage,
            "OFFLINE_BACKFILL_LOCK_PATH",
            Path(directory) / "worker.lock",
        ):
            def replace_status(*_args, **_kwargs):
                usage.write_offline_backfill_status(
                    {
                        "state": "queued",
                        "run_id": "newer-run",
                        "heartbeat_at": now.isoformat(),
                    }
                )
                return False

            with patch.object(
                usage,
                "spawn_offline_backfill_worker",
                side_effect=replace_status,
            ):
                usage.queue_offline_backfill_worker(Path(directory) / "today.json", now)
            saved = usage.read_offline_backfill_status()

        self.assertEqual(saved["run_id"], "newer-run")
        self.assertEqual(saved["state"], "queued")

    def test_worker_scans_each_contiguous_date_group_once(self) -> None:
        now = datetime(2026, 8, 24, 9)
        targets = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 23)]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            usage,
            "OFFLINE_BACKFILL_STATUS_PATH",
            Path(directory) / "status.json",
        ), patch.object(
            usage,
            "OFFLINE_BACKFILL_LOCK_PATH",
            Path(directory) / "worker.lock",
        ), patch.object(
            usage,
            "load_usage_history_for_backfill",
            return_value={"schema": 2, "days": {}},
        ), patch.object(
            usage,
            "opencodex_accounting_migration_dates",
            return_value=set(targets),
        ), patch.object(
            usage,
            "offline_history_dates_to_reconcile",
            return_value=targets,
        ), patch.object(
            usage,
            "load_attribution_ledger",
            return_value={},
        ), patch.object(
            usage,
            "backfill_offline_usage_history",
            side_effect=lambda *args, target_days, **kwargs: {
                "state": "complete",
                "updated_days": len(target_days),
            },
        ) as backfill:
            result = usage.run_offline_backfill_worker(
                Path(directory),
                Path(directory) / "sessions",
                now,
                "run-1",
            )

        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["scanned_days"], 3)
        self.assertEqual(backfill.call_count, 2)
        self.assertEqual(
            [call.kwargs["target_days"] for call in backfill.call_args_list],
            [[date(2026, 8, 20), date(2026, 8, 21)], [date(2026, 8, 23)]],
        )

    def test_backfill_commit_rereads_latest_history(self) -> None:
        target = date(2026, 8, 22)
        today = date(2026, 8, 24)
        stale_history = {"schema": 2, "days": {}}
        latest_history = {
            "schema": 2,
            "days": {today.isoformat(): {"tokens": 999}},
        }
        rebuilt = {
            target.isoformat(): self.history_row(
                200,
                usage.USAGE_ACCOUNTING_SCHEMA,
            )
        }
        history_path: Path
        with tempfile.TemporaryDirectory() as directory, patch.object(
            usage,
            "USAGE_HISTORY_PATH",
            Path(directory) / "usage_history.json",
        ), patch.object(
            usage,
            "build_historical_usage_rows",
            return_value=rebuilt,
        ), patch.object(usage, "refresh_json_backup") as refresh_backup:
            history_path = usage.USAGE_HISTORY_PATH
            usage.write_json_atomic(usage.USAGE_HISTORY_PATH, latest_history)
            result = usage.backfill_offline_usage_history(
                Path(directory),
                Path(directory) / "sessions",
                datetime(2026, 8, 24, 9),
                {},
                history=stale_history,
                target_days=[target],
            )
            saved = json.loads(usage.USAGE_HISTORY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(result["state"], "complete")
        self.assertEqual(saved["days"][today.isoformat()]["tokens"], 999)
        self.assertEqual(saved["days"][target.isoformat()]["tokens"], 200)
        refresh_backup.assert_called_once_with(
            history_path,
            max_age_seconds=0,
        )

    def test_duplicate_with_same_stable_identity_is_still_deduped(self) -> None:
        first = local_event(request_key="request-a")
        duplicate = local_event(request_key="request-a")

        deduped = usage.dedupe_usage_events([first, duplicate])

        self.assertEqual(len(deduped), 1)

    def test_rollout_and_logs2_copy_with_different_keys_are_deduped(self) -> None:
        rollout = local_event(
            BASE,
            session_id="session-a",
            request_key="session-a",
        )
        rollout.usage_provenance = "codex-rollout"
        logs2 = local_event(
            BASE + timedelta(milliseconds=500),
            session_id="session-a",
            request_key="response-a",
        )
        logs2.usage_provenance = "codex-logs2"

        deduped = usage.dedupe_usage_events([rollout, logs2])

        self.assertEqual(len(deduped), 1)

    def test_two_logs2_requests_in_same_session_are_not_cross_source_deduped(self) -> None:
        first = local_event(
            BASE,
            session_id="session-a",
            request_key="response-a",
        )
        first.usage_provenance = "codex-logs2"
        second = local_event(
            BASE + timedelta(milliseconds=500),
            session_id="session-a",
            request_key="response-b",
        )
        second.usage_provenance = "codex-logs2"

        deduped = usage.dedupe_usage_events([first, second])

        self.assertEqual(len(deduped), 2)

    def test_two_rollout_requests_in_same_session_and_second_are_not_deduped(self) -> None:
        first = local_event(
            BASE,
            session_id="session-a",
            request_key="session-a",
        )
        first.usage_provenance = "codex-rollout"
        second = local_event(
            BASE + timedelta(milliseconds=500),
            session_id="session-a",
            request_key="session-a",
        )
        second.usage_provenance = "codex-rollout"

        deduped = usage.dedupe_usage_events([first, second])

        self.assertEqual(len(deduped), 2)

    def test_parallel_rollout_logs2_pairs_are_deduped_one_to_one(self) -> None:
        events = []
        for offset, response_id in ((0, "response-a"), (500, "response-b")):
            rollout = local_event(
                BASE + timedelta(milliseconds=offset),
                session_id="session-a",
                request_key="session-a",
            )
            rollout.usage_provenance = "codex-rollout"
            logs2 = local_event(
                BASE + timedelta(milliseconds=offset + 100),
                session_id="session-a",
                request_key=response_id,
            )
            logs2.usage_provenance = "codex-logs2"
            events.extend((rollout, logs2))

        deduped = usage.dedupe_usage_events(events)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(
            {event.request_key for event in deduped},
            {"response-a", "response-b"},
        )


if __name__ == "__main__":
    unittest.main()
