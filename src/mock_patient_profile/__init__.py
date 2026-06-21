"""
mock-patient-profile: a mock patient-aware morphology profiling workflow.

The package prototypes the computational architecture for a future
patient-derived fibroblast Cell Painting screen using fully open datasets
(BBBC021) and the Way Science profiling stack (CytoTable, CytoDataFrame,
coSMicQC, pycytominer), with Parquet as the canonical storage format.
"""

from . import schema
from .main import show_message
from .schema import (
    CHANNELS,
    COMPARTMENTS,
    DISEASE_GROUPS,
    METADATA_PREFIX,
    SchemaValidationError,
    feature_name,
    partition_columns,
    require_schema,
    single_cell_schema,
    validate_schema,
)
