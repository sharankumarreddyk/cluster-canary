"""Feature engineering — Phase 3.

Public API:
- `build_features` — labeled wide frame → features + train/val/test split.
- `temporal_split` — temporal split of a labeled wide frame.
- Per-family `add_*_features` functions (see submodules) for ad-hoc composition.
"""

from cluster_canary.features.aggregations import add_aggregation_features
from cluster_canary.features.build import (
    FEATURE_COLUMNS,
    build_features,
)
from cluster_canary.features.cpu import add_cpu_features
from cluster_canary.features.lifecycle import add_lifecycle_features
from cluster_canary.features.memory import add_memory_features
from cluster_canary.features.split import (
    SplitMetrics,
    temporal_split,
    write_entity_overlap_opt_in,
)
from cluster_canary.features.temporal import add_temporal_features

__all__ = [
    "FEATURE_COLUMNS",
    "SplitMetrics",
    "add_aggregation_features",
    "add_cpu_features",
    "add_lifecycle_features",
    "add_memory_features",
    "add_temporal_features",
    "build_features",
    "temporal_split",
    "write_entity_overlap_opt_in",
]
