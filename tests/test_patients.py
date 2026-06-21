"""
Tests for the synthetic patient metadata layer.
"""

import polars as pl

from mock_patient_profile import bbbc021, patients, schema
from mock_patient_profile.paths import DataPaths


def _image_table() -> pl.DataFrame:
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
                    "Metadata_Compound": "DMSO",
                    "Metadata_Concentration": 0.0,
                    "Metadata_MoA": "DMSO",
                }
            )
            image_number += 1
    return pl.DataFrame(rows).select(bbbc021.IMAGE_COLUMNS)


def test_build_patient_table_is_deterministic_and_balanced() -> None:
    first = patients.build_patient_table(8, seed=0)
    second = patients.build_patient_table(8, seed=0)
    assert first.equals(second)

    assert first.height == 8
    assert first["Metadata_PatientID"].n_unique() == 8
    # round-robin over 4 disease groups => 2 patients each
    counts = first["Metadata_DiseaseGroup"].value_counts()
    assert set(counts["Metadata_DiseaseGroup"].to_list()) == set(schema.DISEASE_GROUPS)
    assert counts["count"].to_list() == [2, 2, 2, 2]

    # failure type is consistent with disease group
    for group, failure in zip(
        first["Metadata_DiseaseGroup"], first["Metadata_FailureType"]
    ):
        assert failure == patients.FAILURE_TYPE_BY_DISEASE[group]


def test_patient_table_conforms_to_schema(tmp_path) -> None:
    table = patients.build_patient_table(6, seed=1)
    paths = DataPaths(tmp_path).ensure()
    dest = patients.write_patient_table(table, paths)
    written = schema.read_parquet(dest)
    assert schema.validate_schema(written, schema.patient_schema()) == []


def test_assign_patients_attaches_sample_and_patient_metadata() -> None:
    image = _image_table()
    augmented = patients.assign_patients(image, n_patients=4, seed=0)

    # original rows preserved, new metadata columns added
    assert augmented.height == image.height
    for col in (
        "Metadata_SampleID",
        "Metadata_Batch",
        "Metadata_PatientID",
        "Metadata_DiseaseGroup",
        "Metadata_FailureType",
        "Metadata_Age",
        "Metadata_Sex",
    ):
        assert col in augmented.columns
        assert augmented[col].null_count() == 0

    # SampleID is unique per (plate, well); batch equals plate
    per_well = augmented.select(
        ["Metadata_Plate", "Metadata_Well", "Metadata_SampleID", "Metadata_Batch"]
    ).unique()
    assert per_well["Metadata_SampleID"].n_unique() == per_well.height
    assert (per_well["Metadata_Batch"] == per_well["Metadata_Plate"]).all()

    # 8 wells across 4 patients => each patient gets exactly 2 wells
    patient_well_counts = (
        augmented.select(["Metadata_PatientID", "Metadata_SampleID"])
        .unique()
        .group_by("Metadata_PatientID")
        .len()
    )
    assert patient_well_counts["len"].to_list() == [2, 2, 2, 2]


def test_assign_patients_is_deterministic() -> None:
    image = _image_table()
    a = patients.assign_patients(image, n_patients=4, seed=0)
    b = patients.assign_patients(image, n_patients=4, seed=0)
    assert a.equals(b)
