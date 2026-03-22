from feast import Entity
from feast.value_type import ValueType


instance = Entity(
    name="instance_id",
    join_keys=["instance_id"],
    value_type=ValueType.STRING,
    description="Unique UFLP instance identifier",
)

facility = Entity(
    name="facility_key",
    join_keys=["facility_key"],
    value_type=ValueType.STRING,
    description="Synthetic single-key identifier for a facility within an instance",
)

customer = Entity(
    name="customer_key",
    join_keys=["customer_key"],
    value_type=ValueType.STRING,
    description="Synthetic single-key identifier for a customer within an instance",
)
