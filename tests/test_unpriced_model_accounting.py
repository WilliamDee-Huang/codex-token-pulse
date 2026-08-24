import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import client_usage_export as usage


class UnpricedModelAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.online_table = usage._ONLINE_PRICE_TABLE
        self.online_details = usage._ONLINE_PRICE_DETAILS
        self.online_fetched_at = usage._ONLINE_PRICE_FETCHED_AT
        self.online_last_attempt = usage._ONLINE_PRICE_LAST_ATTEMPT_AT
        self.online_cache_path = usage._ONLINE_PRICE_CACHE_PATH
        usage._ONLINE_PRICE_TABLE = {}
        usage._ONLINE_PRICE_DETAILS = {}
        usage._ONLINE_PRICE_FETCHED_AT = None
        usage._ONLINE_PRICE_LAST_ATTEMPT_AT = None
        usage._ONLINE_PRICE_CACHE_PATH = None

    def tearDown(self) -> None:
        usage._ONLINE_PRICE_TABLE = self.online_table
        usage._ONLINE_PRICE_DETAILS = self.online_details
        usage._ONLINE_PRICE_FETCHED_AT = self.online_fetched_at
        usage._ONLINE_PRICE_LAST_ATTEMPT_AT = self.online_last_attempt
        usage._ONLINE_PRICE_CACHE_PATH = self.online_cache_path

    @staticmethod
    def event(model: str, tokens: int = 1_000) -> usage.UsageEvent:
        return usage.UsageEvent(
            when=datetime(2026, 8, 22, 12, 0, 0),
            model=model,
            input_tokens=tokens - 100,
            cached_tokens=0,
            output_tokens=100,
        )

    def test_unknown_model_keeps_tokens_and_reports_unpriced_breakdown(self) -> None:
        bucket = usage.UsageBucket()

        usage.add_codex_event_to_bucket(bucket, self.event("future-model"))
        exported = usage.bucket_to_dict("test", bucket)

        self.assertEqual(bucket.total_tokens, 1_000)
        self.assertEqual(bucket.cost, 0.0)
        self.assertEqual(bucket.unpriced_tokens, 1_000)
        self.assertEqual(bucket.unpriced_models, {"future-model": 1_000})
        self.assertEqual(exported["tokens"], 1_000)
        self.assertEqual(exported["unpriced_tokens"], 1_000)

    def test_nearby_grok_prices_do_not_guess_grok_build_price(self) -> None:
        usage._ONLINE_PRICE_TABLE = {
            "grok-4.6": (3.0, 0.75, 15.0),
            "grok-build-0.1": (2.0, 0.5, 10.0),
        }
        usage._ONLINE_PRICE_DETAILS = {
            "grok-4.6": {
                "input_cost_per_token": 3.0,
                "output_cost_per_token": 15.0,
            },
            "grok-build-0.1": {
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 10.0,
            },
        }

        cost, resolved = usage.estimate_cost_with_resolution(
            "grok-4.6-build",
            1_000,
            0,
            100,
        )

        self.assertFalse(resolved)
        self.assertEqual(cost, 0.0)

    def test_bare_deepseek_price_does_not_override_opencode_exact_fallback(self) -> None:
        details = usage.extract_online_price_details(
            {
                "deepseek/deepseek-v4-pro": {
                    "litellm_provider": "deepseek",
                    "input_cost_per_token": 0.000002,
                    "cache_read_input_token_cost": 0.0000005,
                    "output_cost_per_token": 0.000008,
                }
            }
        )
        usage._ONLINE_PRICE_DETAILS = details
        usage._ONLINE_PRICE_TABLE = usage.extract_online_price_table(
            {
                "deepseek/deepseek-v4-pro": {
                    "litellm_provider": "deepseek",
                    "input_cost_per_token": 0.000002,
                    "cache_read_input_token_cost": 0.0000005,
                    "output_cost_per_token": 0.000008,
                }
            }
        )

        cost, resolved = usage.estimate_cost_with_resolution(
            "opencode-go/deepseek-v4-pro",
            1_000_000,
            0,
            100_000,
        )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.858)
        self.assertIn("deepseek-v4-pro", details)

    def test_arbitrary_prefix_is_not_stripped_for_price_matching(self) -> None:
        usage._ONLINE_PRICE_DETAILS = {
            "deepseek-v4-pro": {
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 8.0,
            }
        }

        cost, resolved = usage.estimate_cost_with_resolution(
            "untrusted/deepseek-v4-pro",
            1_000,
            0,
            100,
        )

        self.assertFalse(resolved)
        self.assertEqual(cost, 0.0)

    def test_bucket_merge_preserves_unpriced_models(self) -> None:
        left = usage.UsageBucket()
        right = usage.UsageBucket()
        right.add_unpriced_model("future-model", 2_500)

        usage.add_bucket(left, right)

        self.assertEqual(left.unpriced_tokens, 2_500)
        self.assertEqual(left.unpriced_models, {"future-model": 2_500})

    def test_cancelled_claude_usage_with_unknown_model_remains_visible(self) -> None:
        bucket = usage.UsageBucket()
        event = usage.ClaudeUsageEvent(
            event_id="claude-unknown",
            when=datetime(2026, 8, 22, 12, 0, 0),
            model="claude-future",
            input_tokens=900,
            output_tokens=100,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

        usage.add_claude_event_to_bucket(bucket, event)

        self.assertEqual(bucket.total_tokens, 1_000)
        self.assertEqual(bucket.unpriced_tokens, 1_000)
        self.assertEqual(bucket.cost, 0.0)

    def test_cockpit_unknown_model_without_upstream_cost_is_unpriced(self) -> None:
        bucket = usage.UsageBucket()

        added = usage.add_cockpit_usage_to_bucket(
            bucket,
            datetime(2026, 8, 22, 12, 0, 0).timestamp() * 1_000,
            "future-model",
            900,
            100,
            1_000,
            0,
            0,
        )

        self.assertTrue(added)
        self.assertEqual(bucket.total_tokens, 1_000)
        self.assertEqual(bucket.cost, 0.0)
        self.assertEqual(bucket.unpriced_tokens, 1_000)
        self.assertEqual(bucket.unpriced_models, {"future-model": 1_000})

    def test_cockpit_upstream_cost_prices_an_unknown_model(self) -> None:
        bucket = usage.UsageBucket()

        added = usage.add_cockpit_usage_to_bucket(
            bucket,
            datetime(2026, 8, 22, 12, 0, 0).timestamp() * 1_000,
            "future-model",
            900,
            100,
            1_000,
            0,
            0.125,
        )

        self.assertTrue(added)
        self.assertAlmostEqual(bucket.cost, 0.125)
        self.assertEqual(bucket.unpriced_tokens, 0)
        self.assertEqual(bucket.unpriced_models, {})

    def test_failed_online_refresh_keeps_valid_cached_price(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "fetched_at": 0,
                        "models": {
                            "deepseek/deepseek-v4-pro": {
                                "litellm_provider": "deepseek",
                                "input_cost_per_token": 0.000002,
                                "output_cost_per_token": 0.000008,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            usage._ONLINE_PRICE_TABLE = None
            usage._ONLINE_PRICE_DETAILS = None
            with (
                patch.object(usage, "MODEL_PRICE_CACHE_PATH", cache_path),
                patch.object(usage.request, "urlopen", side_effect=OSError("offline")),
            ):
                table = usage.load_online_price_table()

        self.assertEqual(table["deepseek-v4-pro"], (2.0, 2.0, 8.0))

    def test_cache_hit_alias_is_used_for_cached_tokens(self) -> None:
        usage._ONLINE_PRICE_DETAILS = {
            "future-cache-model": {
                "input_cost_per_token": 2.0,
                "cache_read_input_token_cost": 0.2,
                "output_cost_per_token": 8.0,
            }
        }
        usage._ONLINE_PRICE_TABLE = {
            "future-cache-model": (2.0, 0.2, 8.0),
        }

        cost, resolved = usage.estimate_cost_with_resolution(
            "future-cache-model",
            1_000_000,
            500_000,
            100_000,
        )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 2.9)

    def test_extract_maps_cache_hit_field_to_cache_read_price(self) -> None:
        details = usage.extract_online_price_details(
            {
                "vendor/future-cache-model": {
                    "litellm_provider": "vendor",
                    "input_cost_per_token": 0.000002,
                    "input_cost_per_token_cache_hit": 0.0000002,
                    "output_cost_per_token": 0.000008,
                }
            }
        )

        self.assertIn("vendor/future-cache-model", details)
        self.assertNotIn("future-cache-model", details)
        self.assertAlmostEqual(
            details["vendor/future-cache-model"]["cache_read_input_token_cost"],
            0.2,
        )

    def test_grok_build_uses_explicit_canonical_pricing_model(self) -> None:
        usage._ONLINE_PRICE_TABLE = {"xai/grok-4.6": (2.0, 0.5, 6.0)}
        usage._ONLINE_PRICE_DETAILS = {
            "xai/grok-4.6": {
                "input_cost_per_token": 2.0,
                "cache_read_input_token_cost": 0.5,
                "output_cost_per_token": 6.0,
            }
        }

        unresolved_cost, unresolved = usage.estimate_cost_with_resolution(
            "grok-4.6-build",
            1_000_000,
            0,
            100_000,
        )
        resolved_cost, resolved = usage.estimate_cost_with_resolution(
            "grok-4.6-build",
            1_000_000,
            0,
            100_000,
            pricing_model="xai/grok-4.6",
        )
        bucket = usage.UsageBucket()
        usage.add_codex_event_to_bucket(
            bucket,
            usage.UsageEvent(
                when=datetime(2026, 8, 22, 12, 0, 0),
                model="grok-4.6-build",
                input_tokens=1_000_000,
                cached_tokens=0,
                output_tokens=100_000,
                pricing_model="xai/grok-4.6",
            ),
        )

        self.assertFalse(unresolved)
        self.assertEqual(unresolved_cost, 0.0)
        self.assertTrue(resolved)
        self.assertAlmostEqual(resolved_cost, 2.6)
        self.assertAlmostEqual(bucket.cost, 2.6)
        self.assertEqual(bucket.unpriced_tokens, 0)
        self.assertEqual(bucket.models, {"grok-4.6-build": 1_100_000})

    def test_grok_local_pricing_model_requires_matching_default(self) -> None:
        self.assertEqual(
            usage.grok_local_pricing_model("grok-4.6-build", default_model="grok-4.6"),
            "xai/grok-4.6",
        )
        self.assertEqual(
            usage.grok_local_pricing_model("grok-4.6-build", default_model="xai/grok-4.6"),
            "xai/grok-4.6",
        )
        self.assertEqual(
            usage.grok_local_pricing_model("grok-4.6-build", default_model=""),
            "",
        )
        self.assertEqual(
            usage.grok_local_pricing_model("grok-4.6-build", default_model="grok-build-0.1"),
            "",
        )
        self.assertEqual(
            usage.grok_local_pricing_model("grok-4.6", default_model="grok-4.6"),
            "",
        )

    def test_unknown_model_miss_refreshes_stale_cache_once(self) -> None:
        payload = {
            "opencode-go/future-pro": {
                "litellm_provider": "opencode-go",
                "input_cost_per_token": 0.00000132,
                "input_cost_per_token_cache_hit": 0.000000044,
                "output_cost_per_token": 0.00000396,
            }
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "fetched_at": 1.0,
                        "models": {
                            "xai/grok-4.6": {
                                "litellm_provider": "xai",
                                "input_cost_per_token": 0.000002,
                                "output_cost_per_token": 0.000006,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            usage._ONLINE_PRICE_TABLE = None
            usage._ONLINE_PRICE_DETAILS = None
            usage._ONLINE_PRICE_CACHE_PATH = None
            class FrozenDateTime(usage.datetime):
                current = 100.0

                @classmethod
                def now(cls, tz=None):
                    class Stamp:
                        def timestamp(self_inner):
                            return cls.current
                    return Stamp()

            with (
                patch.object(usage, "MODEL_PRICE_CACHE_PATH", cache_path),
                patch.object(usage, "MODEL_PRICE_CACHE_SECONDS", 86_400),
                patch.object(usage, "MODEL_PRICE_REFRESH_COOLDOWN_SECONDS", 300),
                patch.object(usage.request, "urlopen", return_value=FakeResponse()) as urlopen,
                patch.object(usage, "datetime", FrozenDateTime),
            ):
                FrozenDateTime.current = 100.0
                first_cost, first_resolved = usage.estimate_cost_with_resolution(
                    "opencode-go/future-pro",
                    1_000_000,
                    500_000,
                    100_000,
                )
                FrozenDateTime.current = 160.0
                second_cost, second_resolved = usage.estimate_cost_with_resolution(
                    "opencode-go/future-flash",
                    1_000,
                    0,
                    100,
                )

            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertTrue(first_resolved)
        self.assertAlmostEqual(first_cost, 1.738)
        self.assertEqual(urlopen.call_count, 2)
        self.assertFalse(second_resolved)
        self.assertEqual(second_cost, 0.0)
        self.assertAlmostEqual(cached["last_attempt_at"], 100.0)
        self.assertIn("opencode-go/future-pro", cached["models"])
        self.assertNotIn("future-pro", cached["models"])

    def test_models_dev_official_routes_keep_exact_provider_aliases(self) -> None:
        payload = {
            "hpc-ai": {
                "id": "hpc-ai",
                "models": {
                    "deepseek/deepseek-v4-pro": {
                        "id": "deepseek/deepseek-v4-pro",
                        "cost": {"input": 1.74, "output": 3.48, "cache_read": 0.145},
                    }
                },
            },
            "deepseek": {
                "id": "deepseek",
                "models": {
                    "deepseek-v4-pro": {
                        "id": "deepseek-v4-pro",
                        "cost": {"input": 0.435, "output": 0.87, "cache_read": 0.003625},
                    }
                },
            },
            "opencode-go": {
                "id": "opencode-go",
                "models": {
                    "deepseek-v4-pro": {
                        "id": "deepseek-v4-pro",
                        "cost": {"input": 0.66, "output": 1.98, "cache_read": 0.022},
                    }
                },
            },
            "xai": {
                "id": "xai",
                "models": {
                    "grok-4.6": {
                        "id": "grok-4.6",
                        "cost": {"input": 2, "output": 6, "cache_read": 0.5},
                    }
                },
            },
        }

        details = usage.extract_online_price_details(payload)
        table = usage.extract_online_price_table(payload)
        usage._ONLINE_PRICE_DETAILS = details
        usage._ONLINE_PRICE_TABLE = table

        self.assertAlmostEqual(details["deepseek-v4-pro"]["input_cost_per_token"], 0.435)
        self.assertAlmostEqual(details["deepseek/deepseek-v4-pro"]["input_cost_per_token"], 0.435)
        self.assertAlmostEqual(details["hpc-ai/deepseek-v4-pro"]["input_cost_per_token"], 1.74)
        self.assertAlmostEqual(details["opencode-go/deepseek-v4-pro"]["input_cost_per_token"], 0.66)
        self.assertAlmostEqual(details["xai/grok-4.6"]["input_cost_per_token"], 2.0)
        self.assertAlmostEqual(details["grok-4.6"]["input_cost_per_token"], 2.0)
        self.assertEqual(table["deepseek-v4-pro"], (0.435, 0.003625, 0.87))
        self.assertEqual(table["hpc-ai/deepseek-v4-pro"], (1.74, 0.145, 3.48))
        self.assertAlmostEqual(table["opencode-go/deepseek-v4-pro"][0], 0.66)
        self.assertAlmostEqual(table["opencode-go/deepseek-v4-pro"][1], 0.022)
        self.assertAlmostEqual(table["opencode-go/deepseek-v4-pro"][2], 1.98)
        cost, resolved = usage.estimate_cost_with_resolution(
            "opencode-go/deepseek-v4-pro",
            1_000_000,
            0,
            0,
        )
        official_cost, official_resolved = usage.estimate_cost_with_resolution(
            "deepseek-v4-pro",
            1_000_000,
            0,
            0,
        )
        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.66)
        self.assertTrue(official_resolved)
        self.assertAlmostEqual(official_cost, 0.435)

    def test_models_dev_bare_price_does_not_replace_opencode_exact_fallback(self) -> None:
        class LiteLLMResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "xai/grok-4.6": {
                        "litellm_provider": "xai",
                        "input_cost_per_token": 0.000002,
                        "cache_read_input_token_cost": 0.0000005,
                        "output_cost_per_token": 0.000006,
                    }
                }).encode("utf-8")

        class ModelsDevResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "deepseek": {
                        "id": "deepseek",
                        "models": {
                            "deepseek-v4-pro": {
                                "id": "deepseek-v4-pro",
                                "cost": {"input": 0.435, "output": 0.87, "cache_read": 0.003625},
                            }
                        },
                    }
                }).encode("utf-8")

        def fake_urlopen(req, timeout=0):
            url = getattr(req, "full_url", None) or getattr(req, "get_full_url", lambda: "")()
            if "models.dev" in str(url):
                return ModelsDevResponse()
            return LiteLLMResponse()

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            usage._ONLINE_PRICE_TABLE = None
            usage._ONLINE_PRICE_DETAILS = None
            usage._ONLINE_PRICE_CACHE_PATH = None
            with (
                patch.object(usage, "MODEL_PRICE_CACHE_PATH", cache_path),
                patch.object(usage.request, "urlopen", side_effect=fake_urlopen),
            ):
                table = usage.load_online_price_table()
                cost, resolved = usage.estimate_cost_with_resolution(
                    "opencode-go/deepseek-v4-pro",
                    1_000_000,
                    0,
                    100_000,
                )
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 0.858)
        self.assertEqual(table["deepseek-v4-pro"], (0.435, 0.003625, 0.87))
        self.assertEqual(table["xai/grok-4.6"], (2.0, 0.5, 6.0))
        self.assertIn("deepseek/deepseek-v4-pro", cached["models"])
        self.assertIn("xai/grok-4.6", cached["models"])

    def test_opencode_go_kimi_uses_exact_official_fallback(self) -> None:
        cost, resolved = usage.estimate_cost_with_resolution(
            "opencode-go/kimi-k3",
            1_000_000,
            500_000,
            100_000,
            when=datetime(2026, 8, 22, 0, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 4.65)
        self.assertIsNone(usage.opencode_go_official_price_details("kimi-k3"))

    def test_opencode_go_deepseek_uses_event_time_peak_boundaries(self) -> None:
        cases = (
            ("opencode-go/deepseek-v4-pro", 0, 2.662),
            ("opencode-go/deepseek-v4-pro", 1, 5.324),
            ("opencode-go/deepseek-v4-pro", 4, 2.662),
            ("opencode-go/deepseek-v4-pro", 6, 5.324),
            ("opencode-go/deepseek-v4-pro", 10, 2.662),
            ("opencode-go/deepseek-v4-flash", 0, 0.887),
            ("opencode-go/deepseek-v4-flash", 1, 1.774),
        )
        for model, hour, expected in cases:
            with self.subTest(model=model, hour=hour):
                cost, resolved = usage.estimate_cost_with_resolution(
                    model,
                    1_000_000,
                    1_000_000,
                    1_000_000,
                    when=datetime(2026, 8, 22, hour, 0, tzinfo=timezone.utc),
                )
                self.assertTrue(resolved)
                self.assertAlmostEqual(cost, expected)

    def test_online_exact_opencode_price_precedes_official_fallback(self) -> None:
        usage._ONLINE_PRICE_DETAILS = {
            "opencode-go/kimi-k3": {
                "input_cost_per_token": 1.0,
                "cache_read_input_token_cost": 0.1,
                "output_cost_per_token": 2.0,
            }
        }
        usage._ONLINE_PRICE_TABLE = {"opencode-go/kimi-k3": (1.0, 0.1, 2.0)}

        cost, resolved = usage.estimate_cost_with_resolution(
            "opencode-go/kimi-k3",
            1_000_000,
            0,
            0,
        )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 1.0)

    def test_online_exact_deepseek_base_price_keeps_peak_multiplier(self) -> None:
        usage._ONLINE_PRICE_DETAILS = {
            "opencode-go/deepseek-v4-pro": {
                "input_cost_per_token": 0.66,
                "cache_read_input_token_cost": 0.022,
                "output_cost_per_token": 1.98,
            }
        }
        usage._ONLINE_PRICE_TABLE = {
            "opencode-go/deepseek-v4-pro": (0.66, 0.022, 1.98)
        }

        cost, resolved = usage.estimate_cost_with_resolution(
            "opencode-go/deepseek-v4-pro",
            1_000_000,
            1_000_000,
            1_000_000,
            when=datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(resolved)
        self.assertAlmostEqual(cost, 5.324)

    def test_successful_source_refresh_keeps_cached_provider_exact_price(self) -> None:
        cached_details = {
            "opencode-go/future-model": {
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 8.0,
            }
        }
        fresh_payload = {
            "xai/grok-4.6": {
                "litellm_provider": "xai",
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000006,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prices.json"
            with (
                patch.object(usage, "MODEL_PRICE_CACHE_PATH", cache_path),
                patch.object(usage, "_fetch_online_price_payload", return_value=fresh_payload),
                patch.object(
                    usage,
                    "_fetch_models_dev_price_payload",
                    side_effect=OSError("offline"),
                ),
            ):
                table = usage._refresh_online_price_table(cached_details, 10.0, 20.0)

        self.assertEqual(table["opencode-go/future-model"], (2.0, 2.0, 8.0))
        self.assertEqual(table["xai/grok-4.6"], (2.0, 2.0, 6.0))

    def test_codex_hourly_cost_uses_event_time_peak_price(self) -> None:
        event = usage.UsageEvent(
            when=datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc),
            model="opencode-go/deepseek-v4-pro",
            input_tokens=1_000_000,
            cached_tokens=0,
            output_tokens=100_000,
        )
        authoritative = usage.UsageBucket()
        usage.add_codex_event_to_bucket(authoritative, event)

        hourly = usage.codex_hourly_from_events([event])[1]

        self.assertAlmostEqual(hourly["cost"], authoritative.cost)
        self.assertAlmostEqual(hourly["cost"], 1.716)

    def test_codex_hourly_cost_uses_explicit_pricing_model(self) -> None:
        usage._ONLINE_PRICE_DETAILS = {
            "xai/grok-4.6": {
                "input_cost_per_token": 2.0,
                "cache_read_input_token_cost": 0.5,
                "output_cost_per_token": 6.0,
            }
        }
        usage._ONLINE_PRICE_TABLE = {"xai/grok-4.6": (2.0, 0.5, 6.0)}
        event = usage.UsageEvent(
            when=datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc),
            model="grok-4.6-build",
            input_tokens=1_000_000,
            cached_tokens=0,
            output_tokens=100_000,
            pricing_model="xai/grok-4.6",
        )
        authoritative = usage.UsageBucket()
        usage.add_codex_event_to_bucket(authoritative, event)

        hourly = usage.codex_hourly_from_events([event])[2]

        self.assertAlmostEqual(hourly["cost"], authoritative.cost)
        self.assertAlmostEqual(hourly["cost"], 2.6)


if __name__ == "__main__":
    unittest.main()
