from __future__ import annotations

from pathlib import Path

from feast import FeatureStore

try:
    from .entities import customer, facility, instance
    from .views import customer_features_view, facility_features_view, instance_features_view
except ImportError:
    from entities import customer, facility, instance
    from views import customer_features_view, facility_features_view, instance_features_view

FEATURE_REPO_DIR = Path(__file__).resolve().parent


def main() -> None:
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    store.apply(
        [
            instance,
            facility,
            customer,
            instance_features_view,
            facility_features_view,
            customer_features_view,
        ]
    )
    print("Feast repository applied successfully.")


if __name__ == "__main__":
    main()
