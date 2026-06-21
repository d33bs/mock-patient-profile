"""
Tests for the CytoDataFrame integration layer.
"""

import pandas as pd
import pyarrow as pa
import pytest

from mock_patient_profile import cytodataframe_io as cdf_io
from mock_patient_profile import patients, schema
from mock_patient_profile.paths import DataPaths


def _single_cell_frame() -> pd.DataFrame:
    rows = []
    for plate in ("PlateA", "PlateB"):
        for well, patient, disease in (
            ("A01", "P001", "Healthy"),
            ("A02", "P002", "Fontan Failure"),
        ):
            for obj in (1, 2, 3):
                rows.append(
                    {
                        "Metadata_Plate": plate,
                        "Metadata_Well": well,
                        "Metadata_ImageNumber": 1,
                        "Metadata_ObjectNumber": obj,
                        "Metadata_PatientID": patient,
                        "Metadata_SampleID": f"{plate}_{well}",
                        "Metadata_DiseaseGroup": disease,
                        "Cells_AreaShape_Area": 100.0 + obj,
                        "Nuclei_Intensity_MeanIntensity_DNA": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_load_single_cells_returns_cytodataframe(tmp_path) -> None:
    from cytodataframe import CytoDataFrame

    paths = DataPaths(tmp_path).ensure()
    frame = _single_cell_frame()
    dest = paths.processed / "single_cell.parquet"
    schema.write_parquet(pa.Table.from_pandas(frame), dest)

    loaded = cdf_io.load_single_cells(paths=paths)
    assert isinstance(loaded, CytoDataFrame)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == len(frame)


def test_select_disease_groups_and_patients() -> None:
    frame = _single_cell_frame()
    healthy = cdf_io.select_disease_groups(frame, "Healthy")
    assert healthy["Metadata_DiseaseGroup"].unique().tolist() == ["Healthy"]

    multi = cdf_io.select_disease_groups(frame, ["Healthy", "Fontan Failure"])
    assert set(multi["Metadata_DiseaseGroup"].unique()) == {
        "Healthy",
        "Fontan Failure",
    }

    one_patient = cdf_io.select_patients(frame, "P001")
    assert one_patient["Metadata_PatientID"].unique().tolist() == ["P001"]


def test_metadata_and_feature_partition() -> None:
    frame = _single_cell_frame()
    meta = cdf_io.metadata_frame(frame)
    feats = cdf_io.feature_frame(frame)
    assert all(col.startswith("Metadata_") for col in meta.columns)
    assert list(feats.columns) == [
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_DNA",
    ]


def test_cell_counts_and_hierarchy_summary() -> None:
    frame = _single_cell_frame()

    per_patient = cdf_io.cell_counts(frame, level="patient")
    assert per_patient["n_cells"].tolist() == [6, 6]  # 2 plates x 3 cells each

    summary = cdf_io.hierarchy_summary(frame)
    assert summary.loc[0, "n_patients"] == 2
    assert summary.loc[0, "n_samples"] == 4  # 2 plates x 2 wells
    assert summary.loc[0, "n_plates"] == 2
    assert summary.loc[0, "n_wells"] == 4
    assert summary.loc[0, "n_cells"] == 12


def test_cell_counts_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        cdf_io.cell_counts(_single_cell_frame(), level="galaxy")


def test_attach_patient_metadata_adds_missing_columns() -> None:
    # a profile-like frame carrying only PatientID, no age/sex
    profiles = pd.DataFrame(
        {
            "Metadata_PatientID": ["P001", "P002"],
            "Cells_AreaShape_Area": [101.0, 95.0],
        }
    )
    patient_table = patients.build_patient_table(4, seed=0).to_pandas()
    joined = cdf_io.attach_patient_metadata(profiles, patient_table)
    assert "Metadata_Age" in joined.columns
    assert "Metadata_DiseaseGroup" in joined.columns
    assert joined["Metadata_Age"].notna().all()
