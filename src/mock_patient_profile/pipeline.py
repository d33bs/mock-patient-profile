"""
End-to-end orchestration of the mock patient-profile workflow.

Wires every stage into a single, reproducible run that realizes the project's
architecture:

    BBBC021 metadata
      -> dev subset (Parquet)
      -> synthetic CellProfiler outputs
      -> CytoTable -> canonical single-cell Parquet
      -> CytoDataFrame (patient-aware load)
      -> coSMicQC (QC report)
      -> pycytominer (normalize / select / aggregate / consensus)
      -> per-patient morphology profile
      -> mock clinical + snRNA-seq tables
      -> DuckDB multi-omic integration

:func:`run_pipeline` downloads the (small) real BBBC021 metadata and runs the
whole flow; :func:`run_from_subset` runs everything downstream of a provided
subset (used by tests to stay offline).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from . import (
    bbbc021,
    cytodataframe_io,
    cytotable_io,
    multiomics,
    patients,
    profiling,
    qc,
    schema,
    synthetic,
)
from .paths import DataPaths, get_data_paths


@dataclass(frozen=True)
class PipelineConfig:
    """Tunable knobs for an end-to-end run.

    Defaults are deliberately small so a first run is fast; raise them for a
    richer dataset once the workflow is proven out.
    """

    n_plates: int = 2
    cells_per_site: int = 15
    n_patients: int = 8
    seed: int = 0
    normalize_method: str = "standardize"
    #: Disease<->plate confounding strength in [0, 1] (0 = balanced/easy).
    disease_plate_confounding: float = 0.0
    #: Signal strengths / feature realism for the generator (None = defaults).
    signal: synthetic.SignalConfig | None = None


def run_from_subset(
    subset: pl.DataFrame,
    paths: DataPaths | None = None,
    *,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """Run every stage downstream of a BBBC021 dev subset.

    Args:
        subset: A site-level BBBC021 dev subset (see
            :func:`mock_patient_profile.bbbc021.build_dev_subset`).
        paths: Data locations. Defaults to :func:`get_data_paths`.
        config: Pipeline configuration. Defaults to :class:`PipelineConfig`.

    Returns:
        A summary dict of counts and output paths.
    """
    config = config or PipelineConfig()
    paths = (paths or get_data_paths()).ensure()

    # patient layer + persisted subset
    bbbc021.write_dev_subset(subset, paths)
    patient_table = patients.build_patient_table(config.n_patients, seed=config.seed)
    patients.write_patient_table(patient_table, paths)
    augmented = patients.assign_patients(
        subset,
        n_patients=config.n_patients,
        seed=config.seed,
        disease_plate_confounding=config.disease_plate_confounding,
    )

    # synthetic CellProfiler outputs -> CytoTable -> canonical single-cell Parquet
    _truth, csv_dir = synthetic.generate_synthetic_dataset(
        augmented,
        paths,
        cells_per_site=config.cells_per_site,
        seed=config.seed,
        signal=config.signal,
    )
    single_cell_path = cytotable_io.build_single_cell_parquet(csv_dir, augmented, paths)

    # CytoDataFrame (patient-aware load) feeds QC + profiling
    single_cells = cytodataframe_io.load_single_cells(single_cell_path)
    qc_tables = qc.qc_report(single_cells, paths)
    profiles = profiling.build_patient_profiles(
        single_cells, paths, method=config.normalize_method
    )

    # mock multi-omics + DuckDB integration
    multiomics.build_multiomic_tables(
        patient_table.to_pandas(), paths, seed=config.seed
    )
    integrated = multiomics.integrate_multiomics(paths)

    selected_features = schema.feature_columns(list(profiles["selected"].columns))
    return {
        "n_plates": int(subset["Metadata_Plate"].n_unique()),
        "n_wells": subset.select(["Metadata_Plate", "Metadata_Well"]).unique().height,
        "n_cells": len(single_cells),
        "n_patients": int(single_cells["Metadata_PatientID"].nunique()),
        "n_features_selected": len(selected_features),
        "n_outlier_flags": int(qc_tables["outliers"]["n_outliers"].sum()),
        "n_integrated_rows": int(integrated.height),
        "outputs": {
            "dev_subset": str(paths.interim / "bbbc021_dev_subset.parquet"),
            "single_cell": str(single_cell_path),
            "patient": str(paths.processed / "patient.parquet"),
            "morphology_profile": str(paths.processed / "morphology_profile.parquet"),
            "clinical": str(paths.processed / "clinical.parquet"),
            "snrna_summary": str(paths.processed / "snrna_summary.parquet"),
            "integrated_patient": str(paths.processed / "integrated_patient.parquet"),
            "qc_report": str(paths.reports / "qc_report.md"),
        },
    }


def run_pipeline(
    paths: DataPaths | None = None,
    *,
    config: PipelineConfig | None = None,
    force_download: bool = False,
    verify: bool = True,
) -> dict[str, object]:
    """Download BBBC021 metadata and run the full end-to-end workflow.

    Args:
        paths: Data locations. Defaults to :func:`get_data_paths`.
        config: Pipeline configuration. Defaults to :class:`PipelineConfig`.
        force_download: Force re-download of BBBC021 metadata.
        verify: Verify downloads against pinned SHA-256 digests.

    Returns:
        A summary dict of counts and output paths.
    """
    config = config or PipelineConfig()
    paths = (paths or get_data_paths()).ensure()
    subset, _ = bbbc021.prepare_dev_subset(
        paths, n_plates=config.n_plates, force=force_download, verify=verify
    )
    return run_from_subset(subset, paths, config=config)
