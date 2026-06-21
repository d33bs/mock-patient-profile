"""
Tests for BBBC021 ingestion and development-subset construction.

These tests are hermetic: metadata frames are built in memory and downloads use
``file://`` URLs. An opt-in smoke test (enabled with
``MOCK_PATIENT_PROFILE_RUN_NETWORK=1``) exercises the real Broad download.
"""

import os

import polars as pl
import pytest

from mock_patient_profile import bbbc021, schema
from mock_patient_profile.paths import DataPaths


def _image_fixture() -> pl.DataFrame:
    """A tiny stand-in for BBBC021_v1_image.csv: 2 plates x 4 wells x 2 sites."""
    rows = []
    image_number = 1
    treatments = [
        ("A01", "DMSO", 0.0),
        ("A02", "taxol", 0.3),
        ("A03", "cytochalasin B", 1.0),
        ("A04", "unknownCpd", 5.0),
    ]
    for plate in ("PlateB", "PlateA"):
        for well, compound, conc in treatments:
            for _site in range(2):
                rows.append(
                    {
                        "TableNumber": 100 + image_number,
                        "ImageNumber": image_number,
                        "Image_Metadata_Plate_DAPI": plate,
                        "Image_Metadata_Well_DAPI": well,
                        "Replicate": 1,
                        "Image_Metadata_Compound": compound,
                        "Image_Metadata_Concentration": conc,
                    }
                )
                image_number += 1
    return pl.DataFrame(rows)


def _moa_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "compound": ["DMSO", "taxol", "cytochalasin B"],
            "concentration": [0.0, 0.3, 1.0],
            "moa": ["DMSO", "Microtubule stabilizers", "Actin disruptors"],
        }
    )


def test_canonicalize_attaches_moa_and_site_index() -> None:
    canon = bbbc021.canonicalize_image_table(_image_fixture(), _moa_fixture())
    assert list(canon.columns) == list(bbbc021.IMAGE_COLUMNS)

    # DMSO controls inherit the DMSO MoA; unknown compound stays null.
    dmso = canon.filter(pl.col("Metadata_Compound") == "DMSO")
    assert dmso["Metadata_MoA"].unique().to_list() == ["DMSO"]
    unknown = canon.filter(pl.col("Metadata_Compound") == "unknownCpd")
    assert unknown["Metadata_MoA"].null_count() == unknown.height

    # Site is a 1-based index within each well.
    one_well = canon.filter(
        (pl.col("Metadata_Plate") == "PlateA") & (pl.col("Metadata_Well") == "A02")
    ).sort("Metadata_Site")
    assert one_well["Metadata_Site"].to_list() == [1, 2]


def test_build_dev_subset_is_deterministic_and_filtered() -> None:
    subset = bbbc021.build_dev_subset(
        _image_fixture(), _moa_fixture(), n_plates=1, moa_only=True
    )
    # n_plates=1 keeps the first plate by sorted name ("PlateA").
    assert subset["Metadata_Plate"].unique().to_list() == ["PlateA"]
    # moa_only drops the unannotated compound, keeps DMSO + 2 treatments.
    assert set(subset["Metadata_Compound"].unique().to_list()) == {
        "DMSO",
        "taxol",
        "cytochalasin B",
    }
    # 3 wells x 2 sites
    assert subset.height == 6
    # sorted by plate, well, site
    assert subset["Metadata_Well"].to_list() == sorted(
        subset["Metadata_Well"].to_list()
    )


def test_validate_image_table() -> None:
    subset = bbbc021.build_dev_subset(_image_fixture(), _moa_fixture(), n_plates=2)
    assert bbbc021.validate_image_table(subset) == []

    broken = subset.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("Metadata_Well"))
        .alias("Metadata_Well")
    )
    problems = bbbc021.validate_image_table(broken)
    assert any("Metadata_Well" in p for p in problems)


def test_write_dev_subset_conforms_to_schema(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    subset = bbbc021.build_dev_subset(_image_fixture(), _moa_fixture(), n_plates=2)
    dest = bbbc021.write_dev_subset(subset, paths)
    assert dest.exists()

    table = schema.read_parquet(dest)
    assert schema.validate_schema(table, schema.image_schema()) == []


def test_download_file_via_file_url_and_caching(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("compound,moa\nDMSO,DMSO\n")
    sha = bbbc021._sha256(source)
    dest = tmp_path / "out" / "downloaded.csv"

    out = bbbc021.download_file(source.as_uri(), dest, sha256=sha)
    assert out == dest
    assert dest.read_text() == source.read_text()

    # Cached path is reused (no .part left behind, same content).
    again = bbbc021.download_file(source.as_uri(), dest, sha256=sha)
    assert again == dest
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_file_integrity_error(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("data\n")
    dest = tmp_path / "downloaded.csv"
    with pytest.raises(bbbc021.BBBC021IntegrityError, match="checksum mismatch"):
        bbbc021.download_file(source.as_uri(), dest, sha256="0" * 64)
    assert not dest.exists()


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MOCK_PATIENT_PROFILE_RUN_NETWORK") != "1",
    reason="set MOCK_PATIENT_PROFILE_RUN_NETWORK=1 to run real BBBC021 download",
)
def test_real_download_and_subset(tmp_path) -> None:
    paths = DataPaths(tmp_path).ensure()
    subset, path = bbbc021.prepare_dev_subset(paths, n_plates=2)
    assert path.exists()
    assert subset.height > 0
    assert subset["Metadata_Plate"].n_unique() == 2
    assert "DMSO" in subset["Metadata_MoA"].unique().to_list()
