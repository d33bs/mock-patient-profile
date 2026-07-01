"""
Tests for the mock multi-omic tables and DuckDB integration join.
"""

import numpy as np
import pandas as pd

from mock_patient_profile import multiomics, patients, schema
from mock_patient_profile.paths import DataPaths

N_PATIENTS = 4
N_CELL_TYPES = len(multiomics.DEFAULT_CELL_TYPES)


def _patient_pd() -> pd.DataFrame:
    return patients.build_patient_table(N_PATIENTS, seed=0).to_pandas()


def test_clinical_table_correlates_with_disease() -> None:
    pt = _patient_pd()
    clinical = multiomics.build_clinical_table(pt, seed=0)
    merged = clinical.merge(
        pt[["Metadata_PatientID", "Metadata_DiseaseGroup"]], on="Metadata_PatientID"
    )
    healthy = merged.loc[merged["Metadata_DiseaseGroup"] == "Healthy"]
    failing = merged.loc[merged["Metadata_DiseaseGroup"] == "Systolic Failure"]
    # failing patients have lower ejection fraction than healthy
    assert healthy["ejection_fraction"].mean() > failing["ejection_fraction"].mean()


def test_snrna_summary_shape_and_fractions() -> None:
    snrna = multiomics.build_snrna_summary_table(_patient_pd(), seed=0)
    assert len(snrna) == N_PATIENTS * N_CELL_TYPES
    per_patient = snrna.groupby("Metadata_PatientID")["fraction_of_sample"].sum()
    assert np.allclose(per_patient.to_numpy(), 1.0, atol=0.01)


def test_write_tables_conform_to_schema(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    multiomics.build_multiomic_tables(_patient_pd(), paths, seed=0)

    clinical = schema.read_parquet(paths.processed / "clinical.parquet")
    assert schema.validate_schema(clinical, schema.clinical_schema()) == []
    snrna = schema.read_parquet(paths.processed / "snrna_summary.parquet")
    assert schema.validate_schema(snrna, schema.snrna_summary_schema()) == []


def test_integrate_multiomics_joins_to_one_row_per_patient(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    patient_table = patients.build_patient_table(N_PATIENTS, seed=0)
    patients.write_patient_table(patient_table, paths)
    multiomics.build_multiomic_tables(patient_table.to_pandas(), paths, seed=0)

    # minimal morphology profile (one row per patient + 2 features)
    morph = patient_table.to_pandas().copy()
    morph["Cells_AreaShape_Area"] = [1.0, 2.0, 3.0, 4.0]
    morph["Nuclei_Intensity_MeanIntensity_DNA"] = [0.1, 0.2, 0.3, 0.4]
    morph.to_parquet(paths.processed / "morphology_profile.parquet", index=False)

    integrated = multiomics.integrate_multiomics(paths, write=True)
    assert integrated.height == N_PATIENTS
    columns = integrated.columns
    assert "ejection_fraction" in columns
    assert "fibroblast_fraction" in columns
    assert "Cells_AreaShape_Area" in columns
    # morphology metadata excluded to avoid clashing with patient columns
    assert columns.count("Metadata_DiseaseGroup") == 1
    assert (paths.processed / "integrated_patient.parquet").exists()
