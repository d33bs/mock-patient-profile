"""
BBBC021 metadata ingestion and development-subset construction.

BBBC021 (Caie et al. 2010; Ljosa et al. 2013) is a public MCF7 Cell Painting
benchmark distributed by the Broad Bioimage Benchmark Collection. This module
provides a reproducible, hash-verified download of its small metadata CSVs and
builds a deliberately tiny, deterministic *development subset* (a few plates,
restricted to mechanism-of-action-annotated treatments plus DMSO controls).

Only the lightweight metadata is fetched here: the full image set is large and
unnecessary for prototyping the computational architecture. The single-cell
morphology features are synthesized downstream (see
:mod:`mock_patient_profile.synthetic`) and anchored to this real metadata.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import polars as pl

from . import schema
from .paths import DataPaths, get_data_paths

#: Base URL for the Broad-hosted BBBC021 metadata files.
BBBC021_BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC021/"

#: Logical name -> filename for each metadata CSV.
BBBC021_FILES: dict[str, str] = {
    "image": "BBBC021_v1_image.csv",
    "compound": "BBBC021_v1_compound.csv",
    "moa": "BBBC021_v1_moa.csv",
}

#: Pinned SHA-256 digests for reproducibility. A mismatch means the upstream
#: data changed and results may not reproduce, so the download fails loudly.
BBBC021_SHA256: dict[str, str] = {
    "image": "7fd8a9363970548736a03eba3e14bd2b737c967f0e0fdf436bc91004676f50e7",
    "compound": "e4b6868ea77eb1cef67cfe75dd792a2074ffde609c79d69c42deec131b1b88ec",
    "moa": "cddb6c6b72c043fd25f3a41d84624d4817927a1d76fe87dc829674181dd62719",
}

#: Canonical column order of the site-level image table (matches image_schema).
IMAGE_COLUMNS: tuple[str, ...] = tuple(schema.image_schema().names)

#: Metadata columns that must never be null in a valid image table.
_REQUIRED_NON_NULL = (
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_ImageNumber",
    "Metadata_Compound",
)

_DOWNLOAD_CHUNK = 1 << 20


class BBBC021IntegrityError(RuntimeError):
    """Raised when a downloaded file fails its SHA-256 integrity check."""


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: str | Path,
    *,
    sha256: str | None = None,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download a URL to ``dest`` atomically, with optional integrity checking.

    The file is streamed to a ``.part`` sibling and only moved into place after
    an optional SHA-256 verification passes, so partial or corrupt downloads are
    never observed by the rest of the workflow. An existing valid file is reused
    unless ``force`` is set.

    Args:
        url: Source URL (``https://`` for real data, ``file://`` in tests).
        dest: Destination path.
        sha256: Expected hex digest. When provided, both a cached and a freshly
            downloaded file are verified against it.
        force: Re-download even if a valid cached file exists.
        timeout: Socket timeout in seconds.

    Returns:
        The destination path.

    Raises:
        BBBC021IntegrityError: If the downloaded bytes do not match ``sha256``.
    """
    dest = Path(dest)
    if dest.exists() and not force and (sha256 is None or _sha256(dest) == sha256):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with (
        urllib.request.urlopen(url, timeout=timeout) as response,
        part.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=_DOWNLOAD_CHUNK)

    if sha256 is not None:
        actual = _sha256(part)
        if actual != sha256:
            part.unlink(missing_ok=True)
            raise BBBC021IntegrityError(
                f"checksum mismatch for {url}: expected {sha256}, got {actual}"
            )

    part.replace(dest)
    return dest


def download_bbbc021_metadata(
    paths: DataPaths | None = None,
    *,
    force: bool = False,
    verify: bool = True,
) -> dict[str, Path]:
    """Download the three BBBC021 metadata CSVs into the raw data directory.

    Args:
        paths: Target data locations. Defaults to :func:`get_data_paths`.
        force: Re-download even if cached copies exist.
        verify: Verify each file against its pinned SHA-256 digest.

    Returns:
        Mapping of logical name (``image``/``compound``/``moa``) to local path.
    """
    paths = paths or get_data_paths()
    paths.raw.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for key, name in BBBC021_FILES.items():
        downloaded[key] = download_file(
            BBBC021_BASE_URL + name,
            paths.raw / name,
            sha256=BBBC021_SHA256[key] if verify else None,
            force=force,
        )
    return downloaded


def load_bbbc021_metadata(
    paths: DataPaths | None = None,
) -> dict[str, pl.DataFrame]:
    """Read the downloaded BBBC021 metadata CSVs as Polars frames.

    Args:
        paths: Data locations. Defaults to :func:`get_data_paths`.

    Returns:
        Mapping of logical name to a raw (un-canonicalized) Polars frame.
    """
    paths = paths or get_data_paths()
    return {key: pl.read_csv(paths.raw / name) for key, name in BBBC021_FILES.items()}


def canonicalize_image_table(
    image: pl.DataFrame,
    moa: pl.DataFrame,
) -> pl.DataFrame:
    """Rename BBBC021 image metadata to canonical columns and attach MoA.

    Args:
        image: Raw ``BBBC021_v1_image.csv`` frame.
        moa: Raw ``BBBC021_v1_moa.csv`` frame (compound, concentration, moa).

    Returns:
        A site-level Polars frame with the canonical :data:`IMAGE_COLUMNS`.
        ``Metadata_Site`` is a 1-based index of the field of view within a well,
        and ``Metadata_MoA`` is null for treatments without a known mechanism.
    """
    canon = image.rename(
        {
            "TableNumber": "Metadata_TableNumber",
            "ImageNumber": "Metadata_ImageNumber",
            "Image_Metadata_Plate_DAPI": "Metadata_Plate",
            "Image_Metadata_Well_DAPI": "Metadata_Well",
            "Replicate": "Metadata_Replicate",
            "Image_Metadata_Compound": "Metadata_Compound",
            "Image_Metadata_Concentration": "Metadata_Concentration",
        }
    ).with_columns(
        pl.col("Metadata_TableNumber").cast(pl.Utf8),
        pl.col("Metadata_ImageNumber").cast(pl.Int32),
        pl.col("Metadata_Replicate").cast(pl.Int32),
        pl.col("Metadata_Concentration").cast(pl.Float64),
    )

    # Join MoA on (compound, concentration); round concentration to avoid float
    # mismatch. DMSO controls carry the "DMSO" MoA in the upstream table.
    moa_small = (
        moa.rename({"moa": "Metadata_MoA"})
        .with_columns(_conc_key=pl.col("concentration").round(6))
        .select(["compound", "_conc_key", "Metadata_MoA"])
    )
    canon = (
        canon.with_columns(_conc_key=pl.col("Metadata_Concentration").round(6))
        .join(
            moa_small,
            left_on=["Metadata_Compound", "_conc_key"],
            right_on=["compound", "_conc_key"],
            how="left",
        )
        .drop("_conc_key")
    )

    canon = canon.with_columns(
        Metadata_Site=pl.col("Metadata_ImageNumber")
        .rank(method="ordinal")
        .over(["Metadata_Plate", "Metadata_Well"])
        .cast(pl.Int32)
    )
    return canon.select(IMAGE_COLUMNS)


def build_dev_subset(
    image: pl.DataFrame,
    moa: pl.DataFrame,
    *,
    n_plates: int = 3,
    moa_only: bool = True,
) -> pl.DataFrame:
    """Build a small, deterministic development subset of BBBC021.

    Args:
        image: Raw image-metadata frame.
        moa: Raw MoA frame.
        n_plates: Number of plates to keep (first ``n_plates`` by sorted plate
            name, for determinism).
        moa_only: Keep only MoA-annotated treatments (including DMSO controls).

    Returns:
        A site-level Polars frame conforming to :func:`schema.image_schema`,
        sorted by plate, well, and site.
    """
    canon = canonicalize_image_table(image, moa)
    plates = sorted(canon["Metadata_Plate"].unique().to_list())[:n_plates]
    subset = canon.filter(pl.col("Metadata_Plate").is_in(plates))
    if moa_only:
        subset = subset.filter(pl.col("Metadata_MoA").is_not_null())
    return subset.sort(["Metadata_Plate", "Metadata_Well", "Metadata_Site"])


def validate_image_table(table: pl.DataFrame) -> list[str]:
    """Validate a canonicalized image table and return any problems found.

    Checks structural data quality (required columns, non-null keys, sane
    ranges). Arrow-type conformance is enforced separately at write time via
    :func:`schema.cast_to_schema`.

    Args:
        table: A canonicalized image table.

    Returns:
        A list of human-readable problems; empty means the table is valid.
    """
    problems: list[str] = []

    missing = [col for col in IMAGE_COLUMNS if col not in table.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
        return problems

    if table.height == 0:
        problems.append("image table is empty")
        return problems

    for col in _REQUIRED_NON_NULL:
        null_count = table[col].null_count()
        if null_count:
            problems.append(f"column '{col}' has {null_count} null value(s)")

    if table["Metadata_Site"].min() < 1:
        problems.append("Metadata_Site must be >= 1")
    if table["Metadata_Concentration"].min() < 0:
        problems.append("Metadata_Concentration must be >= 0")

    return problems


def write_dev_subset(
    subset: pl.DataFrame,
    paths: DataPaths | None = None,
) -> Path:
    """Write the development subset to canonical Parquet in ``interim``.

    Args:
        subset: A site-level subset frame.
        paths: Data locations. Defaults to :func:`get_data_paths`.

    Returns:
        The path of the written Parquet file.
    """
    paths = paths or get_data_paths()
    dest = paths.interim / "bbbc021_dev_subset.parquet"
    return schema.write_parquet(subset.to_arrow(), dest, schema=schema.image_schema())


def prepare_dev_subset(
    paths: DataPaths | None = None,
    *,
    n_plates: int = 3,
    force: bool = False,
    verify: bool = True,
) -> tuple[pl.DataFrame, Path]:
    """Download metadata, build the dev subset, validate, and persist it.

    This is the one-call entrypoint used by the end-to-end pipeline.

    Args:
        paths: Data locations (created if needed). Defaults to
            :func:`get_data_paths`.
        n_plates: Number of plates to include.
        force: Force re-download of metadata.
        verify: Verify downloaded files against pinned digests.

    Returns:
        The ``(subset_frame, parquet_path)`` pair.

    Raises:
        ValueError: If the constructed subset fails validation.
    """
    paths = (paths or get_data_paths()).ensure()
    download_bbbc021_metadata(paths, force=force, verify=verify)
    metadata = load_bbbc021_metadata(paths)
    subset = build_dev_subset(metadata["image"], metadata["moa"], n_plates=n_plates)
    problems = validate_image_table(subset)
    if problems:
        raise ValueError("invalid BBBC021 dev subset: " + "; ".join(problems))
    path = write_dev_subset(subset, paths)
    return subset, path
