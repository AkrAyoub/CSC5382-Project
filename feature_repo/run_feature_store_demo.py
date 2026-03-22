from __future__ import annotations

from pathlib import Path

from feast import FeatureStore
import pandas as pd


FEATURE_REPO_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_TIMESTAMP = "2026-03-21T00:00:00Z"


def build_demo_entity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["cap71", "cap101", "capa"],
            "event_timestamp": pd.to_datetime([DEFAULT_EVENT_TIMESTAMP] * 3),
        }
    )


def main() -> None:
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    print("=== Historical retrieval demo ===")
    historical = store.get_historical_features(
        entity_df=build_demo_entity_df(),
        features=[
            "instance_features_view:facility_count_m",
            "instance_features_view:customer_count_n",
            "instance_features_view:avg_fixed_cost",
            "instance_features_view:avg_assignment_cost",
        ],
    ).to_df()

    print(historical)


if __name__ == "__main__":
    main()
