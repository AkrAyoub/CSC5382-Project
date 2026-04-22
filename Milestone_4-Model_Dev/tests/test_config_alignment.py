from __future__ import annotations

import unittest

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.utils.config import load_yaml_config


class ConfigAlignmentTests(unittest.TestCase):
    def test_train_best_candidate_exists_in_comparison_candidates(self) -> None:
        train_config = load_yaml_config(M4_ROOT / "configs" / "train_best_model.yaml")
        compare_config = load_yaml_config(M4_ROOT / "configs" / "compare_models.yaml")

        selected_candidate = train_config["candidate"]["name"]
        compare_candidates = compare_config["comparison"]["candidate_names"]

        self.assertIn(selected_candidate, compare_candidates)

    def test_train_and_compare_candidate_overrides_match(self) -> None:
        train_config = load_yaml_config(M4_ROOT / "configs" / "train_best_model.yaml")
        compare_config = load_yaml_config(M4_ROOT / "configs" / "compare_models.yaml")

        candidate_name = train_config["candidate"]["name"]
        compare_override = compare_config["comparison"][candidate_name]

        for key, value in train_config["candidate"].items():
            if key == "name":
                continue
            self.assertEqual(compare_override[key], value)


if __name__ == "__main__":
    unittest.main()
