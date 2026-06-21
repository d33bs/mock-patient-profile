"""
Tests for the CytoTable integration layer.

The metadata re-attachment logic is tested without CytoTable (using a hand-built
CytoTable-like Parquet). A separate, slower test marked ``integration`` runs the
real ``cytotable.convert`` on synthetic CellProfiler CSVs.
"""

import polars as pl
import pytest

from mock_patient_profile import (
    bbbc021,
    cytotable_io,
    patients,
    schema,
    synthetic,
)
from mock_patient_profile.paths import DataPaths


def test_feature_columns_from_cytotable_filters_bookkeeping() -> None:
    columns = [
        "Metadata_ImageNumber",
        "Metadata_ObjectNumber",
        "Image_FileName_DNA",
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_DNA",
        "Cells_Number_Object_Number",
        "Cytoplasm_Number_Object_Number",
    ]
    features = cytotable_io.feature_columns_from_cytotable(columns)
    assert features == [
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_DNA",
    ]


def _augmented_table() -> pl.DataFrame:
    rows = []
    image_number = 1
    for plate in ("PlateA", "PlateB"):
        for well in ("A01", "A02"):
            rows.append(
                {
                    "Metadata_TableNumber": str(image_number),
                    "Metadata_ImageNumber": image_number,
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": 1,
                    "Metadata_Replicate": 1,
                    "Metadata_Compound": "taxol",
                    "Metadata_Concentration": 0.3,
                    "Metadata_MoA": "Microtubule stabilizers",
                }
            )
            image_number += 1
    image = pl.DataFrame(rows).select(bbbc021.IMAGE_COLUMNS)
    return patients.assign_patients(image, n_patients=4, seed=0)


def _cytotable_like_parquet(path, image_numbers) -> None:
    """Write a minimal CytoTable-style single-cell Parquet (2 cells per image)."""
    rows = []
    for image_number in image_numbers:
        for obj in (1, 2):
            rows.append(
                {
                    # int64 join keys to exercise casting
                    "Metadata_ImageNumber": image_number,
                    "Metadata_ObjectNumber": obj,
                    "Image_FileName_DNA": "x.tif",
                    "Cells_AreaShape_Area": 100.0 + obj,
                    "Nuclei_Intensity_MeanIntensity_DNA": 0.5,
                    "Cytoplasm_Texture_Contrast_Actin": 3.0,
                    "Cells_Number_Object_Number": obj,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_to_canonical_single_cells_attaches_metadata(tmp_path) -> None:
    augmented = _augmented_table()
    ct_path = tmp_path / "cytotable.parquet"
    _cytotable_like_parquet(ct_path, augmented["Metadata_ImageNumber"].to_list())

    dest = cytotable_io.to_canonical_single_cells(
        ct_path, augmented, tmp_path / "single_cell.parquet"
    )
    table = schema.read_parquet(dest)

    features = [
        "Cells_AreaShape_Area",
        "Cytoplasm_Texture_Contrast_Actin",
        "Nuclei_Intensity_MeanIntensity_DNA",
    ]
    assert schema.validate_schema(table, schema.single_cell_schema(features)) == []

    result = pl.from_arrow(table)
    # 4 images x 2 cells
    assert result.height == augmented.height * 2
    # rich metadata attached, no nulls in patient/treatment columns
    for col in ("Metadata_DiseaseGroup", "Metadata_PatientID", "Metadata_MoA"):
        assert result[col].null_count() == 0
    # bookkeeping + filename columns dropped
    assert "Cells_Number_Object_Number" not in result.columns
    assert "Image_FileName_DNA" not in result.columns


@pytest.mark.integration
def test_build_single_cell_parquet_with_real_cytotable(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    augmented = _augmented_table()
    _cells, csv_dir = synthetic.generate_synthetic_dataset(
        augmented, paths, cells_per_site=5, seed=0
    )

    dest = cytotable_io.build_single_cell_parquet(csv_dir, augmented, paths)
    table = schema.read_parquet(dest)

    expected = schema.single_cell_schema(synthetic.canonical_feature_names())
    assert schema.validate_schema(table, expected) == []
    assert table.num_rows == augmented.height * 5

    result = pl.from_arrow(table)
    assert result["Metadata_DiseaseGroup"].null_count() == 0
    assert result["Metadata_PatientID"].n_unique() == 4
