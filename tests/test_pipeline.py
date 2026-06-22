"""
End-to-end pipeline test (offline).

Exercises the full workflow downstream of a provided BBBC021 dev subset, so it
runs the real CytoTable / coSMicQC / pycytominer / DuckDB stack without any
network access. Marked ``integration`` because it is comparatively slow.
"""

from pathlib import Path

import polars as pl
import pytest

from mock_patient_profile import bbbc021, pipeline, schema, synthetic
from mock_patient_profile.paths import DataPaths


def _dev_subset() -> pl.DataFrame:
    """A site-level subset: 2 plates x 3 wells x 2 sites."""
    treatments = [
        ("A01", "DMSO", 0.0, "DMSO"),
        ("A02", "taxol", 0.3, "Microtubule stabilizers"),
        ("A03", "cytochalasin B", 1.0, "Actin disruptors"),
    ]
    rows = []
    image_number = 1
    for plate in ("PlateA", "PlateB"):
        for well, compound, conc, moa in treatments:
            for site in (1, 2):
                rows.append(
                    {
                        "Metadata_TableNumber": str(image_number),
                        "Metadata_ImageNumber": image_number,
                        "Metadata_Plate": plate,
                        "Metadata_Well": well,
                        "Metadata_Site": site,
                        "Metadata_Replicate": 1,
                        "Metadata_Compound": compound,
                        "Metadata_Concentration": conc,
                        "Metadata_MoA": moa,
                    }
                )
                image_number += 1
    return pl.DataFrame(rows).select(bbbc021.IMAGE_COLUMNS)


@pytest.mark.integration
def test_run_from_subset_end_to_end(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    config = pipeline.PipelineConfig(cells_per_site=6, n_patients=4, seed=0)

    summary = pipeline.run_from_subset(_dev_subset(), paths, config=config)

    # every advertised output artifact exists
    for key in (
        "dev_subset",
        "single_cell",
        "patient",
        "morphology_profile",
        "clinical",
        "snrna_summary",
        "integrated_patient",
        "qc_report",
    ):
        assert Path(summary["outputs"][key]).exists(), key

    # 2 plates x 3 wells x 2 sites x 6 cells = 72 cells
    assert summary["n_cells"] == 72
    assert summary["n_patients"] == 4
    assert summary["n_features_selected"] >= 1

    # the canonical single-cell parquet conforms to schema
    single_cells = schema.read_parquet(summary["outputs"]["single_cell"])
    expected = schema.single_cell_schema(synthetic.canonical_feature_names())
    assert schema.validate_schema(single_cells, expected) == []

    # one integrated multi-omic row per patient
    integrated = schema.read_parquet(summary["outputs"]["integrated_patient"])
    assert integrated.num_rows == 4
    assert "ejection_fraction" in integrated.column_names
    assert "fibroblast_fraction" in integrated.column_names
