"""
Tests for the canonical schema module.
"""

import pyarrow as pa
import pytest

from mock_patient_profile import schema


def test_is_metadata_column() -> None:
    assert schema.is_metadata_column("Metadata_PatientID")
    assert not schema.is_metadata_column("Cells_AreaShape_Area")


def test_partition_columns_preserves_order() -> None:
    names = [
        "Metadata_Plate",
        "Cells_AreaShape_Area",
        "Metadata_Well",
        "Nuclei_Intensity_MeanIntensity_DNA",
    ]
    metadata, features = schema.partition_columns(names)
    assert metadata == ["Metadata_Plate", "Metadata_Well"]
    assert features == [
        "Cells_AreaShape_Area",
        "Nuclei_Intensity_MeanIntensity_DNA",
    ]
    assert schema.metadata_columns(names) == metadata
    assert schema.feature_columns(names) == features


def test_feature_name_with_and_without_channel() -> None:
    assert (
        schema.feature_name("Cells", "Intensity", "MeanIntensity", "DNA")
        == "Cells_Intensity_MeanIntensity_DNA"
    )
    assert schema.feature_name("Nuclei", "AreaShape", "Area") == "Nuclei_AreaShape_Area"


def test_single_cell_schema_types_and_dedup() -> None:
    features = ["Cells_AreaShape_Area", "Nuclei_Intensity_MeanIntensity_DNA"]
    sc_schema = schema.single_cell_schema(features)

    # metadata columns precede feature columns
    assert sc_schema.names[: len(schema.SINGLE_CELL_KEY)] == list(
        schema.SINGLE_CELL_KEY
    )
    # de-duplicated even though Metadata_Plate appears in key + sample metadata
    assert sc_schema.names.count("Metadata_Plate") == 1
    # features are float64 and present
    for feat in features:
        assert sc_schema.field(feat).type == pa.float64()
    # canonical metadata types are honored
    assert sc_schema.field("Metadata_Age").type == pa.int32()
    assert sc_schema.field("Metadata_PatientID").type == pa.string()


def test_single_cell_schema_rejects_metadata_prefixed_features() -> None:
    with pytest.raises(ValueError, match="must not use"):
        schema.single_cell_schema(["Metadata_SneakyFeature"])


def test_profile_schema_has_cell_count_and_features() -> None:
    prof = schema.profile_schema(["Cells_AreaShape_Area"])
    assert prof.field("Metadata_CellCount").type == pa.int32()
    assert prof.field("Cells_AreaShape_Area").type == pa.float64()
    assert "Metadata_PatientID" in prof.names


def test_fixed_table_schemas() -> None:
    assert schema.patient_schema().field("Metadata_DiseaseGroup").type == pa.string()
    assert schema.clinical_schema().field("ejection_fraction").type == pa.float64()
    assert schema.clinical_schema().field("on_beta_blocker").type == pa.bool_()
    assert schema.snrna_summary_schema().field("cell_type").type == pa.string()


def test_metadata_fields_rejects_unknown_column() -> None:
    with pytest.raises(KeyError, match="Unknown canonical metadata"):
        schema.single_cell_schema(["Cells_AreaShape_Area"], metadata_fields=("Nope",))


def _toy_table() -> pa.Table:
    return pa.table(
        {
            "Metadata_PatientID": pa.array(["P01"], type=pa.string()),
            "Metadata_DiseaseGroup": pa.array(["Healthy"], type=pa.string()),
            "Metadata_FailureType": pa.array(["None"], type=pa.string()),
            "Metadata_Age": pa.array([42], type=pa.int32()),
            "Metadata_Sex": pa.array(["F"], type=pa.string()),
        }
    )


def test_validate_schema_passes_for_conforming_table() -> None:
    assert schema.validate_schema(_toy_table(), schema.patient_schema()) == []


def test_validate_schema_detects_missing_and_type_mismatch() -> None:
    table = (
        _toy_table()
        .drop_columns(["Metadata_Sex"])
        .set_column(3, "Metadata_Age", pa.array([42], type=pa.int64()))
    )
    problems = schema.validate_schema(table, schema.patient_schema())
    assert any("missing required column 'Metadata_Sex'" in p for p in problems)
    assert any("Metadata_Age" in p and "expected" in p for p in problems)


def test_validate_schema_extra_columns_toggle() -> None:
    table = _toy_table().append_column(
        "Cells_AreaShape_Area", pa.array([1.0], type=pa.float64())
    )
    assert any(
        "unexpected columns" in p
        for p in schema.validate_schema(table, schema.patient_schema())
    )
    assert (
        schema.validate_schema(table, schema.patient_schema(), allow_extra_columns=True)
        == []
    )


def test_require_schema_raises() -> None:
    bad = _toy_table().drop_columns(["Metadata_Sex"])
    with pytest.raises(schema.SchemaValidationError, match="missing required column"):
        schema.require_schema(bad, schema.patient_schema())
