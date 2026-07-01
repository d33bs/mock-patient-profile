"""
Tests for the coSMicQC-based QC reporting workflow.
"""

import numpy as np
import pandas as pd

from mock_patient_profile import qc
from mock_patient_profile.paths import DataPaths


def _single_cells() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_per = 80
    frames = []
    for plate, disease, patient in (
        ("PlateA", "Healthy", "P001"),
        ("PlateB", "Fontan Failure", "P002"),
    ):
        df = pd.DataFrame(
            {
                "Metadata_Plate": plate,
                "Metadata_Batch": plate,
                "Metadata_Well": "A01",
                "Metadata_SampleID": f"{plate}_A01",
                "Metadata_PatientID": patient,
                "Metadata_DiseaseGroup": disease,
                "Nuclei_AreaShape_Area": rng.normal(400, 30, n_per),
                "Cells_AreaShape_Area": rng.normal(800, 60, n_per),
                "Nuclei_Intensity_IntegratedIntensity_DNA": rng.normal(350, 30, n_per),
                "Nuclei_Intensity_MeanIntensity_DNA": rng.normal(0.5, 0.05, n_per),
                "Cells_Intensity_MeanIntensity_Actin": rng.normal(0.5, 0.05, n_per),
            }
        )
        frames.append(df)
    cells = pd.concat(frames, ignore_index=True)

    # plate/batch effect on a non-outlier feature -> should top the drift report
    cells.loc[
        cells["Metadata_Plate"] == "PlateB", "Nuclei_Intensity_MeanIntensity_DNA"
    ] += 0.3
    # clear outliers in a threshold feature
    cells.loc[:3, "Nuclei_AreaShape_Area"] = 5000.0
    # missingness in a non-threshold feature
    cells.loc[0, "Cells_Intensity_MeanIntensity_Actin"] = np.nan
    return cells


def test_missingness_detects_injected_nan() -> None:
    miss = qc.missingness(_single_cells())
    row = miss.loc[miss["feature"] == "Cells_Intensity_MeanIntensity_Actin"]
    assert int(row["n_missing"].iloc[0]) == 1


def test_feature_drift_ranks_batch_shifted_feature_first() -> None:
    drift = qc.feature_drift(_single_cells(), by="Metadata_Batch")
    assert drift.iloc[0]["feature"] == "Nuclei_Intensity_MeanIntensity_DNA"
    assert drift.iloc[0]["drift_score"] > 0


def test_detect_outliers_flags_injected_outliers() -> None:
    labeled, summary = qc.detect_outliers(_single_cells())
    assert "cqc_is_outlier" in labeled.columns
    # the 4 injected giant nuclei must be flagged
    assert labeled["cqc_is_outlier"].sum() >= 4
    area_row = summary.loc[summary["feature"] == "Nuclei_AreaShape_Area"]
    assert int(area_row["n_outliers"].iloc[0]) >= 4


def test_qc_report_writes_outputs(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    report = qc.qc_report(_single_cells(), paths, write=True)

    expected_keys = {
        "cell_counts_per_well",
        "cell_counts_per_plate",
        "cell_counts_per_disease",
        "missingness",
        "feature_drift_by_batch",
        "plate_effects",
        "outliers",
    }
    assert set(report) == expected_keys
    assert (paths.reports / "qc_report.md").exists()
    assert (paths.reports / "qc_missingness.parquet").exists()

    md = (paths.reports / "qc_report.md").read_text()
    assert "Single-cell QC report" in md
    assert "coSMicQC" in md
