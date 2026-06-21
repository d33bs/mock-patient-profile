"""
Tests for the synthetic CellProfiler-style feature generator.
"""

import polars as pl

from mock_patient_profile import schema, synthetic


def _augmented_table(disease: str = "Healthy") -> pl.DataFrame:
    rows = []
    image_number = 1
    for plate in ("PlateA", "PlateB"):
        for well in ("A01", "A02"):
            rows.append(
                {
                    "Metadata_ImageNumber": image_number,
                    "Metadata_Plate": plate,
                    "Metadata_Well": well,
                    "Metadata_Site": 1,
                    "Metadata_MoA": "DMSO",
                    "Metadata_DiseaseGroup": disease,
                }
            )
            image_number += 1
    return pl.DataFrame(rows)


def test_feature_catalog_sizes() -> None:
    # 3 AreaShape + 3 channels * (2 intensity + 1 texture) = 12 per compartment
    assert len(synthetic.compartment_measurements()) == 12
    # 12 measurements * 3 compartments = 36 canonical features
    names = synthetic.canonical_feature_names()
    assert len(names) == 36
    metadata, features = schema.partition_columns(names)
    assert metadata == []
    assert all(name.startswith(schema.COMPARTMENTS) for name in features)


def test_simulate_single_cells_shape_and_positivity() -> None:
    table = _augmented_table()
    cells = synthetic.simulate_single_cells(table, cells_per_site=5, seed=0)

    assert cells.height == table.height * 5
    # ObjectNumber is 1..cells_per_site within each image
    per_image = cells.group_by("Metadata_ImageNumber").agg(
        pl.col("Metadata_ObjectNumber").min().alias("lo"),
        pl.col("Metadata_ObjectNumber").max().alias("hi"),
    )
    assert per_image["lo"].unique().to_list() == [1]
    assert per_image["hi"].unique().to_list() == [5]

    features = synthetic.canonical_feature_names()
    feature_values = cells.select(features)
    assert all(col in cells.columns for col in features)
    # all morphology features are strictly positive
    assert feature_values.min_horizontal().min() > 0


def test_simulate_is_deterministic() -> None:
    table = _augmented_table()
    a = synthetic.simulate_single_cells(table, cells_per_site=4, seed=7)
    b = synthetic.simulate_single_cells(table, cells_per_site=4, seed=7)
    assert a.equals(b)


def test_disease_group_changes_features() -> None:
    # Same seed, sites, MoA, and plates => identical noise; only the disease
    # signal differs, so the feature means must differ between groups.
    healthy = synthetic.simulate_single_cells(
        _augmented_table("Healthy"), cells_per_site=20, seed=0
    )
    failing = synthetic.simulate_single_cells(
        _augmented_table("Systolic Failure"), cells_per_site=20, seed=0
    )
    features = synthetic.canonical_feature_names()
    healthy_means = healthy.select(features).mean().to_numpy().ravel()
    failing_means = failing.select(features).mean().to_numpy().ravel()
    # at least some features shift meaningfully between disease groups
    rel_diff = abs(healthy_means - failing_means) / healthy_means
    assert (rel_diff > 0.05).any()


def test_write_cellprofiler_csvs(tmp_path) -> None:
    table = _augmented_table()
    cells = synthetic.simulate_single_cells(table, cells_per_site=3, seed=0)
    written = synthetic.write_cellprofiler_csvs(cells, table, tmp_path)

    assert set(written) == {"Image", "Cells", "Cytoplasm", "Nuclei"}
    for path in written.values():
        assert path.exists()

    image = pl.read_csv(written["Image"])
    assert "Count_Cells" in image.columns
    assert image.height == table.height
    assert image["Count_Cells"].unique().to_list() == [3]

    cells_csv = pl.read_csv(written["Cells"])
    # compartment CSV columns are unprefixed measurements + keys
    assert "AreaShape_Area" in cells_csv.columns
    assert "Cells_AreaShape_Area" not in cells_csv.columns
    assert {"ImageNumber", "ObjectNumber", "Number_Object_Number"} <= set(
        cells_csv.columns
    )

    cyto_csv = pl.read_csv(written["Cytoplasm"])
    assert {"Parent_Cells", "Parent_Nuclei"} <= set(cyto_csv.columns)
