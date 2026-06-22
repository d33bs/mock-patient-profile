# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # mock-patient-profile: end-to-end walkthrough
#
# This notebook runs the full mock workflow and inspects the outputs. It mirrors
# the single command `mock-patient-profile run`, but step-by-step so each stage
# is visible.
#
# `jupytext` keeps this `.py` file paired with a notebook view via the
# `pyproject.toml` configuration, so it can be reviewed as a script or opened as
# a notebook.

import polars as pl

from mock_patient_profile import cytodataframe_io, pipeline
from mock_patient_profile.paths import get_data_paths

# Resolve the (git-ignored) data directory used by the workflow.
paths = get_data_paths().ensure()

# ## Run the pipeline
#
# Downloads the small, hash-pinned BBBC021 metadata and runs the whole flow:
# dev subset -> synthetic CellProfiler outputs -> CytoTable -> CytoDataFrame ->
# coSMicQC -> pycytominer -> per-patient profile -> DuckDB multi-omic join.

config = pipeline.PipelineConfig(n_plates=2, cells_per_site=15, n_patients=8)
summary = pipeline.run_pipeline(paths, config=config)
summary

# ## Inspect the canonical single cells (via CytoDataFrame)

single_cells = cytodataframe_io.load_single_cells(paths=paths)
print(cytodataframe_io.hierarchy_summary(single_cells))
single_cells.head()

# ## Per-patient morphology profile
#
# The pycytominer consensus profile — one row per patient.

morphology = pl.read_parquet(paths.processed / "morphology_profile.parquet")
morphology.head()

# ## Integrated multi-omic table
#
# Patient metadata + clinical fields + snRNA-seq summary + morphology features,
# joined with DuckDB. Note how clinical severity tracks the disease group.

integrated = pl.read_parquet(paths.processed / "integrated_patient.parquet")
integrated.select(
    [
        "Metadata_PatientID",
        "Metadata_DiseaseGroup",
        "ejection_fraction",
        "nyha_class",
        "fibroblast_fraction",
    ]
)

# ## QC report
#
# The coSMicQC Markdown summary is written alongside the QC tables.

print((paths.reports / "qc_report.md").read_text())
