from __future__ import annotations

import unittest

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.models.model_registry import list_supported_candidate_names, resolve_candidate_spec


class ModelRegistryTests(unittest.TestCase):
    def test_default_candidates_include_expected_symbolic_variants(self) -> None:
        supported = list_supported_candidate_names()
        self.assertIn("deterministic_baseline", supported)
        self.assertIn("llm_token_prompt_v0", supported)
        self.assertIn("llm_robust_prompt_v1", supported)
        self.assertIn("llm_fine_tuned", supported)

    def test_override_updates_candidate_metadata(self) -> None:
        spec = resolve_candidate_spec(
            "llm_fine_tuned",
            {"enabled": True, "model_name": "groq/fine-tuned-demo", "max_tokens": 900},
        )

        self.assertTrue(spec.enabled)
        self.assertEqual(spec.model_name, "groq/fine-tuned-demo")
        self.assertEqual(spec.max_tokens, 900)


if __name__ == "__main__":
    unittest.main()
