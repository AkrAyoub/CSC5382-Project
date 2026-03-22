from pathlib import Path

from feast import FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float32, Int64, String

try:
    from .entities import customer, facility, instance
except ImportError:
    from entities import customer, facility, instance

ROOT = Path(__file__).resolve().parent.parent


def build_file_source(file_name: str) -> FileSource:
    """Point Feast at one of the generated feature parquet files."""
    return FileSource(
        path=str(ROOT / "data" / "features" / file_name),
        timestamp_field="event_timestamp",
    )


instance_source = build_file_source("instance_features.parquet")
facility_source = build_file_source("facility_features.parquet")
customer_source = build_file_source("customer_features.parquet")

instance_features_view = FeatureView(
    name="instance_features_view",
    entities=[instance],
    ttl=None,
    schema=[
        Field(name="facility_count_m", dtype=Int64),
        Field(name="customer_count_n", dtype=Int64),
        Field(name="facility_customer_ratio", dtype=Float32),
        Field(name="total_fixed_cost", dtype=Float32),
        Field(name="avg_fixed_cost", dtype=Float32),
        Field(name="min_fixed_cost", dtype=Float32),
        Field(name="max_fixed_cost", dtype=Float32),
        Field(name="std_fixed_cost", dtype=Float32),
        Field(name="total_assignment_cost_entries", dtype=Int64),
        Field(name="avg_assignment_cost", dtype=Float32),
        Field(name="min_assignment_cost", dtype=Float32),
        Field(name="max_assignment_cost", dtype=Float32),
        Field(name="std_assignment_cost", dtype=Float32),
        Field(name="fixed_cost_range", dtype=Float32),
        Field(name="assignment_cost_range", dtype=Float32),
    ],
    source=instance_source,
    online=True,
)

facility_features_view = FeatureView(
    name="facility_features_view",
    entities=[facility],
    ttl=None,
    schema=[
        Field(name="instance_id", dtype=String),
        Field(name="facility_id", dtype=Int64),
        Field(name="fixed_cost", dtype=Float32),
        Field(name="normalized_fixed_cost_minmax", dtype=Float32),
        Field(name="fixed_cost_zscore", dtype=Float32),
        Field(name="fixed_cost_rank_ascending", dtype=Int64),
        Field(name="avg_assignment_cost_from_facility", dtype=Float32),
        Field(name="min_assignment_cost_from_facility", dtype=Float32),
        Field(name="max_assignment_cost_from_facility", dtype=Float32),
        Field(name="std_assignment_cost_from_facility", dtype=Float32),
    ],
    source=facility_source,
    online=True,
)

customer_features_view = FeatureView(
    name="customer_features_view",
    entities=[customer],
    ttl=None,
    schema=[
        Field(name="instance_id", dtype=String),
        Field(name="customer_id", dtype=Int64),
        Field(name="min_assignment_cost", dtype=Float32),
        Field(name="max_assignment_cost", dtype=Float32),
        Field(name="avg_assignment_cost", dtype=Float32),
        Field(name="std_assignment_cost", dtype=Float32),
        Field(name="assignment_cost_range", dtype=Float32),
        Field(name="nearest_facility_id", dtype=Int64),
        Field(name="nearest_facility_cost", dtype=Float32),
    ],
    source=customer_source,
    online=True,
)
