"""
CytoTable integration: CellProfiler outputs -> canonical single-cell Parquet.

CytoTable performs the heavy lifting of joining the per-compartment CellProfiler
tables (Cells / Cytoplasm / Nuclei) into one single-cell table. Its default
``cellprofiler_csv`` join keeps only image filenames from the Image table, so
this module re-attaches the rich plate/well/treatment/patient metadata by
``Metadata_ImageNumber`` and writes a table conforming to the canonical
:func:`mock_patient_profile.schema.single_cell_schema`.

The thin wrappers here are intentionally generic (they work on any CellProfiler
output, not just the synthetic mock) so they could be lifted into CytoTable or a
CytoTable-adjacent helper library.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema
from .paths import DataPaths, get_data_paths

#: Suffix marking CytoTable's per-compartment object-number bookkeeping columns.
_BOOKKEEPING_SUFFIX = "Number_Object_Number"

#: Compartment column prefixes (e.g. ``Cells_``) used to spot feature columns.
_COMPARTMENT_PREFIXES = tuple(f"{compartment}_" for compartment in schema.COMPARTMENTS)


def convert_cellprofiler_csvs(
    csv_dir: str | Path,
    dest_path: str | Path,
    *,
    preset: str = "cellprofiler_csv",
) -> Path:
    """Convert a directory of CellProfiler CSVs to a joined single-cell Parquet.

    A thin, dependency-isolating wrapper around :func:`cytotable.convert`
    (imported lazily so the heavy CytoTable/parsl stack only loads when used).

    Args:
        csv_dir: Directory containing ``Image.csv`` and the compartment CSVs.
        dest_path: Destination Parquet path (overwritten if present).
        preset: CytoTable preset name.

    Returns:
        The destination path of the raw CytoTable Parquet output.
    """
    # Imported lazily so the heavy CytoTable/parsl/anndata stack only loads when
    # a conversion actually runs, keeping `import mock_patient_profile` light.
    from cytotable import convert

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    convert(
        source_path=str(csv_dir),
        dest_path=str(dest),
        dest_datatype="parquet",
        source_datatype="csv",
        preset=preset,
    )
    return dest


def feature_columns_from_cytotable(columns: list[str]) -> list[str]:
    """Pick real morphology-feature columns from CytoTable output column names.

    Keeps ``Compartment_*`` columns while dropping CytoTable's
    ``*_Number_Object_Number`` bookkeeping columns. Returned sorted for a stable
    canonical column order.

    Args:
        columns: All column names from a CytoTable single-cell table.

    Returns:
        Sorted list of morphology-feature column names.
    """
    return sorted(
        name
        for name in columns
        if name.startswith(_COMPARTMENT_PREFIXES)
        and not name.endswith(_BOOKKEEPING_SUFFIX)
    )


def to_canonical_single_cells(
    cytotable_parquet: str | Path,
    augmented_table: pl.DataFrame,
    dest_path: str | Path,
) -> Path:
    """Clean a CytoTable output and attach metadata -> canonical single cells.

    Drops CytoTable bookkeeping columns, normalizes the join keys, left-joins the
    site-level patient/treatment metadata on ``Metadata_ImageNumber``, and writes
    a Parquet file validated against :func:`schema.single_cell_schema`.

    Args:
        cytotable_parquet: Path to the raw CytoTable single-cell Parquet.
        augmented_table: Site-level table with patient metadata attached (see
            :func:`mock_patient_profile.patients.assign_patients`).
        dest_path: Destination canonical Parquet path.

    Returns:
        The destination path.
    """
    cells = pl.from_arrow(schema.read_parquet(cytotable_parquet))
    cells = cells.with_columns(
        pl.col("Metadata_ImageNumber").cast(pl.Int32),
        pl.col("Metadata_ObjectNumber").cast(pl.Int32),
    )

    features = feature_columns_from_cytotable(cells.columns)
    cells = cells.select(["Metadata_ImageNumber", "Metadata_ObjectNumber", *features])

    joined = cells.join(augmented_table, on="Metadata_ImageNumber", how="left")
    canonical_schema = schema.single_cell_schema(features)
    final = joined.select(list(canonical_schema.names))
    return schema.write_parquet(final.to_arrow(), dest_path, schema=canonical_schema)


def build_single_cell_parquet(
    csv_dir: str | Path,
    augmented_table: pl.DataFrame,
    paths: DataPaths | None = None,
    *,
    preset: str = "cellprofiler_csv",
) -> Path:
    """Run CytoTable on CellProfiler CSVs and produce canonical single cells.

    Args:
        csv_dir: Directory of CellProfiler CSVs.
        augmented_table: Site-level patient-augmented metadata table.
        paths: Data locations. Defaults to :func:`get_data_paths`.
        preset: CytoTable preset name.

    Returns:
        Path to ``processed/single_cell.parquet``.
    """
    paths = paths or get_data_paths()
    raw = convert_cellprofiler_csvs(
        csv_dir, paths.interim / "cytotable_single_cell.parquet", preset=preset
    )
    dest = paths.processed / "single_cell.parquet"
    return to_canonical_single_cells(raw, augmented_table, dest)
