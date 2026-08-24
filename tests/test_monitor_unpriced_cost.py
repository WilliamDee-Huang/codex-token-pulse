import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import monitor


class MonitorUnpricedCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.live_cache = monitor._LIVE_MODEL_PRICE_CACHE
        monitor._LIVE_MODEL_PRICE_CACHE = None

    def tearDown(self) -> None:
        monitor._LIVE_MODEL_PRICE_CACHE = self.live_cache

    @staticmethod
    def sample(route: str = "codex-local") -> dict[str, int | str]:
        return {
            "route": route,
            "total_tokens": 1_100,
            "input_tokens": 1_000,
            "cached_tokens": 200,
            "output_tokens": 100,
        }

    def test_unknown_non_grok_model_does_not_borrow_today_average(self) -> None:
        with patch.object(monitor, "_load_live_model_prices", return_value={}):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                self.sample(),
                "future-model",
                fallback_cost_per_token=0.01,
            )

        self.assertFalse(resolved)
        self.assertEqual(cost, 0.0)
        self.assertEqual(
            monitor.estimate_live_usage_cost(
                self.sample(),
                "future-model",
                fallback_cost_per_token=0.01,
            ),
            0.0,
        )

    def test_grok_build_does_not_reuse_nearby_model_price(self) -> None:
        prices = {
            "grok-4.6": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }
        with patch.object(monitor, "_load_live_model_prices", return_value=prices):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                self.sample("grok-local"),
                "grok-4.6-build",
            )

        self.assertFalse(resolved)
        self.assertEqual(cost, 0.0)

    def test_grok_build_live_cost_uses_explicit_pricing_model(self) -> None:
        prices = {
            "xai/grok-4.6": {
                "input_cost_per_token": 0.000002,
                "cache_read_input_token_cost": 0.0000005,
                "output_cost_per_token": 0.000006,
            },
            "grok-build-0.1": {
                "input_cost_per_token": 0.000009,
                "output_cost_per_token": 0.000027,
            },
        }
        usage = dict(self.sample("grok-local"))
        usage["pricing_model"] = "xai/grok-4.6"
        with patch.object(monitor, "_load_live_model_prices", return_value=prices):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                usage,
                "grok-4.6-build",
            )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0023)

    def test_deepseek_exact_cache_price_supports_opencode_route(self) -> None:
        payload = {
            "models": {
                "opencode-go/deepseek-v4-pro": {
                    "input_cost_per_token": 0.000002,
                    "cache_read_input_token_cost": 0.0000005,
                    "output_cost_per_token": 0.000008,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(monitor, "MODEL_PRICE_CACHE_JSON", cache_path):
                cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                    self.sample(),
                    "opencode-go/deepseek-v4-pro",
                )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0025)

    def test_bare_deepseek_cache_does_not_override_opencode_fallback(self) -> None:
        prices = {
            "deepseek-v4-pro": {
                "input_cost_per_token": 0.000002,
                "cache_read_input_token_cost": 0.0000005,
                "output_cost_per_token": 0.000008,
            }
        }
        usage = dict(self.sample())
        usage["when"] = "2026-08-22T00:00:00+00:00"
        with patch.object(monitor, "_load_live_model_prices", return_value=prices):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                usage,
                "opencode-go/deepseek-v4-pro",
            )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0007304)

    def test_opencode_deepseek_live_cost_uses_event_time_peak_price(self) -> None:
        usage = dict(self.sample())
        usage["when"] = "2026-08-22T01:00:00+00:00"
        with patch.object(monitor, "_load_live_model_prices", return_value={}):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                usage,
                "opencode-go/deepseek-v4-flash",
            )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0004868)

    def test_cached_opencode_base_price_keeps_live_peak_multiplier(self) -> None:
        prices = {
            "opencode-go/deepseek-v4-pro": {
                "input_cost_per_token": 0.00000066,
                "cache_read_input_token_cost": 0.000000022,
                "output_cost_per_token": 0.00000198,
            }
        }
        usage = dict(self.sample())
        usage["when"] = "2026-08-22T01:00:00+00:00"
        with patch.object(monitor, "_load_live_model_prices", return_value=prices):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                usage,
                "opencode-go/deepseek-v4-pro",
            )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0014608)

    def test_untrusted_prefix_is_not_stripped(self) -> None:
        prices = {
            "deepseek-v4-pro": {
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000008,
            }
        }
        with patch.object(monitor, "_load_live_model_prices", return_value=prices):
            cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                self.sample(),
                "untrusted/deepseek-v4-pro",
            )

        self.assertFalse(resolved)
        self.assertEqual(cost, 0.0)

    def test_unreadable_price_file_keeps_last_valid_runtime_cache(self) -> None:
        cached_prices = {
            "gpt-test": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        }
        monitor._LIVE_MODEL_PRICE_CACHE = ("cached", 1, cached_prices)
        with patch.object(
            monitor,
            "MODEL_PRICE_CACHE_JSON",
            Path("Z:/missing/token-pulse-prices.json"),
        ):
            loaded = monitor._load_live_model_prices()

        self.assertIs(loaded, cached_prices)

    def test_live_summary_includes_uncovered_unpriced_model_breakdown(self) -> None:
        hour = datetime.now(monitor.CN_TZ).hour
        app = monitor.FloatingMonitorApp.__new__(monitor.FloatingMonitorApp)
        app.state = monitor.MonitorState(
            loading=False,
            updated_at=0.0,
            mode="local",
            usage_source="local",
            today_requests=3,
            today_tokens=150,
            client_usage={
                "tokens": 100,
                "requests": 2,
                "input_tokens": 60,
                "cached_input_tokens": 30,
                "output_tokens": 10,
                "unpriced_tokens": 20,
                "unpriced_models": {"old-model": 20},
                "dashboard": {
                    "hourly_today": [
                        {
                            "hour": value,
                            "tokens": 100 if value == hour else 0,
                            "requests": 2 if value == hour else 0,
                            "cost": 0.0,
                        }
                        for value in range(24)
                    ]
                },
            },
        )
        app._live_usage_overlay = {
            "base_today_tokens": 100,
            "base_authoritative_tokens": 100,
            "tokens": 50,
            "input_tokens": 40,
            "cached_input_tokens": 0,
            "output_tokens": 10,
            "unpriced_tokens": 50,
            "unpriced_models": {"future-model": 50},
            "base_hourly": {},
            "hourly": {},
        }

        summary = app._usage_range_summary("24h")

        self.assertEqual(summary["unpriced_tokens"], 70)
        self.assertEqual(
            summary["unpriced_models"],
            {"old-model": 20, "future-model": 50},
        )

    def test_live_cache_hit_alias_is_used_for_cached_tokens(self) -> None:
        payload = {
            "models": {
                "vendor/future-cache-model": {
                    "input_cost_per_token": 0.000002,
                    "input_cost_per_token_cache_hit": 0.0000002,
                    "output_cost_per_token": 0.000008,
                }
            }
        }
        usage = {
            "total_tokens": 1_600,
            "input_tokens": 1_500,
            "cached_tokens": 500,
            "output_tokens": 100,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(monitor, "MODEL_PRICE_CACHE_JSON", cache_path):
                cost, resolved = monitor.estimate_live_usage_cost_with_resolution(
                    usage,
                    "vendor/future-cache-model",
                )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.0029)

    def test_live_non_whitelisted_exact_key_does_not_create_bare_alias(self) -> None:
        payload = {
            "models": {
                "vendor/future-cache-model": {
                    "input_cost_per_token": 0.000002,
                    "output_cost_per_token": 0.000008,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(monitor, "MODEL_PRICE_CACHE_JSON", cache_path):
                exact_cost, exact_resolved = monitor.estimate_live_usage_cost_with_resolution(
                    self.sample(),
                    "vendor/future-cache-model",
                )
                alias_cost, alias_resolved = monitor.estimate_live_usage_cost_with_resolution(
                    self.sample(),
                    "future-cache-model",
                )

        self.assertTrue(exact_resolved)
        self.assertGreater(exact_cost, 0)
        self.assertFalse(alias_resolved)
        self.assertEqual(alias_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
