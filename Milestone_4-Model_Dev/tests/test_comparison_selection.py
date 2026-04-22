from __future__ import annotations

import unittest

import pandas as pd

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.reporting.comparison_reports import build_selection_payload, order_results_df


class ComparisonSelectionTests(unittest.TestCase):
    def test_ordering_uses_validation_metrics_and_ignores_baseline_for_selection(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "candidate_name": "deterministic_baseline",
                    "candidate_kind": "deterministic_baseline",
                    "split": "val",
                    "success_rate": 1.0,
                    "exact_match_rate": 1.0,
                    "mean_gap_vs_baseline_pct": 0.0,
                },
                {
                    "candidate_name": "llm_token_prompt_v0",
                    "candidate_kind": "llm",
                    "split": "val",
                    "success_rate": 0.60,
                    "exact_match_rate": 0.40,
                    "mean_gap_vs_baseline_pct": 0.25,
                },
                {
                    "candidate_name": "llm_robust_prompt_v1",
                    "candidate_kind": "llm",
                    "split": "val",
                    "success_rate": 0.60,
                    "exact_match_rate": 0.55,
                    "mean_gap_vs_baseline_pct": 0.10,
                },
            ]
        )

        ordered = order_results_df(df)

        self.assertEqual(ordered.iloc[0]["candidate_name"], "llm_robust_prompt_v1")
        payload = build_selection_payload(ordered.iloc[0].to_dict())
        self.assertEqual(payload["selected_candidate_name"], "llm_robust_prompt_v1")


if __name__ == "__main__":
    unittest.main()
