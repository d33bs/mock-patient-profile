"""
Tests for the pycytominer profiling workflow.
"""

import numpy as np
import pandas as pd
import polars as pl

from mock_patient_profile import bbbc021, patients, profiling, schema, synthetic
from mock_patient_profile.paths import DataPaths

N_PATIENTS = 4


def _single_cells() -> pd.DataFrame:
    rows = []
    image_number = 1
    for plate in ("PlateA", "PlateB"):
        for well in ("A01", "A02", "A03", "A04"):
            rows.append(
                {
                    "Metadata_TableNumber": str(image_number),
                    "Metadata_ImageNumber": image_number,
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": 1,
                    "Metadata_Replicate": 1,
                    "Metadata_Compound": "DMSO" if well == "A01" else "taxol",
                    "Metadata_Concentration": 0.0 if well == "A01" else 0.3,
                    "Metadata_MoA": "DMSO"
                    if well == "A01"
                    else "Microtubule stabilizers",
                }
            )
            image_number += 1
    image = pl.DataFrame(rows).select(bbbc021.IMAGE_COLUMNS)
    augmented = patients.assign_patients(image, n_patients=N_PATIENTS, seed=0)
    return synthetic.simulate_single_cells(
        augmented, cells_per_site=10, seed=0
    ).to_pandas()


def test_aggregate_to_wells_counts_objects() -> None:
    cells = _single_cells()
    wells = profiling.aggregate_to_wells(cells)
    # 2 plates x 4 wells = 8 well profiles
    assert len(wells) == 8
    assert "Metadata_Object_Count" in wells.columns
    assert wells["Metadata_Object_Count"].unique().tolist() == [10]


def test_normalize_standardize_centers_features() -> None:
    cells = _single_cells()
    wells = profiling.aggregate_to_wells(cells)
    norm = profiling.normalize_profiles(wells, method="standardize")
    features = schema.feature_columns(list(norm.columns))
    means = norm[features].mean().to_numpy()
    assert np.allclose(means, 0.0, atol=1e-6)


def test_feature_select_reduces_features() -> None:
    cells = _single_cells()
    wells = profiling.aggregate_to_wells(cells)
    norm = profiling.normalize_profiles(wells)
    selected = profiling.select_features(norm)
    n_before = len(schema.feature_columns(list(norm.columns)))
    n_after = len(schema.feature_columns(list(selected.columns)))
    assert 1 <= n_after <= n_before


def test_consensus_one_row_per_patient() -> None:
    cells = _single_cells()
    wells = profiling.aggregate_to_wells(cells)
    consensus = profiling.consensus_profiles(profiling.normalize_profiles(wells))
    assert len(consensus) == N_PATIENTS
    assert consensus["Metadata_PatientID"].nunique() == N_PATIENTS


def test_build_patient_profiles_writes_outputs(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    result = profiling.build_patient_profiles(_single_cells(), paths, write=True)

    assert set(result) == {
        "well_profiles",
        "normalized",
        "selected",
        "patient_profiles",
    }
    assert len(result["patient_profiles"]) == N_PATIENTS
    assert (paths.processed / "morphology_profile.parquet").exists()
    assert (paths.processed / "well_profiles.parquet").exists()

    profile = schema.read_parquet(paths.processed / "morphology_profile.parquet")
    assert "Metadata_PatientID" in profile.column_names
