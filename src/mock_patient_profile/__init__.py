"""
mock-patient-profile: a mock patient-aware morphology profiling workflow.

The package prototypes the computational architecture for a future
patient-derived fibroblast Cell Painting screen using fully open datasets
(BBBC021) and the Way Science profiling stack (CytoTable, CytoDataFrame,
coSMicQC, pycytominer), with Parquet as the canonical storage format.
"""

from . import (
    bbbc021,
    cytodataframe_io,
    cytotable_io,
    paths,
    patients,
    profiling,
    qc,
    schema,
    synthetic,
)
from .bbbc021 import build_dev_subset, download_bbbc021_metadata, prepare_dev_subset
from .cytodataframe_io import cell_counts, hierarchy_summary, load_single_cells
from .cytotable_io import build_single_cell_parquet, convert_cellprofiler_csvs
from .main import show_message
from .paths import DataPaths, get_data_paths
from .patients import assign_patients, build_patient_table
from .profiling import aggregate_to_wells, build_patient_profiles, normalize_profiles
from .qc import detect_outliers, qc_report
from .schema import (
    CHANNELS,
    COMPARTMENTS,
    DISEASE_GROUPS,
    METADATA_PREFIX,
    SchemaValidationError,
    cast_to_schema,
    feature_name,
    image_schema,
    partition_columns,
    read_parquet,
    require_schema,
    single_cell_schema,
    validate_schema,
    write_parquet,
)
from .synthetic import (
    canonical_feature_names,
    generate_synthetic_dataset,
    simulate_single_cells,
)
