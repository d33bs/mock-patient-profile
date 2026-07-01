"""
Canonical Parquet schema for the mock patient-profile workflow.

This module is the single source of truth for the project's data model. Every
other module reads and writes tables that conform to the conventions defined
here, and Parquet is treated as the canonical on-disk format throughout.

Design goals
------------
- **Vendor neutral.** Schemas are expressed with :mod:`pyarrow` only, so they
  can be consumed by Polars, DuckDB, pandas, or Arrow directly without any
  proprietary tooling.
- **Profiling-ecosystem compatible.** Metadata columns use the ``Metadata_``
  prefix convention shared by `pycytominer`, `CytoTable`, and related Way
  Science tooling. Any column *not* prefixed with ``Metadata_`` is treated as a
  morphology feature. This makes tables produced here drop-in compatible with
  ``pycytominer`` normalization/aggregation and lets the helpers below double as
  a small, portable schema layer that could be upstreamed.
- **Patient aware.** The metadata vocabulary models the
  Patient -> Sample -> Plate -> Well -> Site -> Cell hierarchy required to
  prototype a patient-derived fibroblast screen.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: Prefix marking a column as metadata rather than a morphology feature. This
#: matches the convention used by pycytominer and CytoTable.
METADATA_PREFIX = "Metadata_"

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

#: Synthetic disease groups for the mock patient layer (see Milestone 6). These
#: mirror a single-ventricle congenital heart disease cohort that the future
#: fibroblast screen is expected to study.
DISEASE_GROUPS: tuple[str, ...] = (
    "Healthy",
    "Stable SV",
    "Fontan Failure",
    "Systolic Failure",
)

#: Heart-failure phenotype associated with each disease group.
FAILURE_TYPES: tuple[str, ...] = (
    "None",
    "Compensated",
    "Fontan",
    "Systolic",
)

#: Reported sex values used in the synthetic patient metadata.
SEXES: tuple[str, ...] = ("F", "M")

#: CellProfiler compartments produced for each object in the mock features.
COMPARTMENTS: tuple[str, ...] = ("Cells", "Cytoplasm", "Nuclei")

#: Imaging channels. BBBC021 is a three-channel Cell Painting-style assay
#: (DNA / Hoechst, beta-tubulin, and F-actin).
CHANNELS: tuple[str, ...] = ("DNA", "Tubulin", "Actin")


# ---------------------------------------------------------------------------
# Canonical metadata fields and their Arrow types
# ---------------------------------------------------------------------------

#: The full universe of canonical metadata columns mapped to their Arrow type.
#: Individual tables select the subset relevant to them. Keeping a single typed
#: registry guarantees a column means the same thing everywhere it appears.
CANONICAL_METADATA_TYPES: dict[str, pa.DataType] = {
    # patient / sample identity
    "Metadata_PatientID": pa.string(),
    "Metadata_SampleID": pa.string(),
    "Metadata_DiseaseGroup": pa.string(),
    "Metadata_FailureType": pa.string(),
    "Metadata_Age": pa.int32(),
    "Metadata_Sex": pa.string(),
    # plate / well / site location
    "Metadata_Batch": pa.string(),
    "Metadata_Plate": pa.string(),
    "Metadata_Well": pa.string(),
    "Metadata_Site": pa.int32(),
    "Metadata_Replicate": pa.int32(),
    # object identity (CellProfiler-style)
    "Metadata_TableNumber": pa.string(),
    "Metadata_ImageNumber": pa.int32(),
    "Metadata_ObjectNumber": pa.int32(),
    # perturbation (BBBC021 treatment)
    "Metadata_Compound": pa.string(),
    "Metadata_Concentration": pa.float64(),
    "Metadata_MoA": pa.string(),
    # bookkeeping for aggregated profiles
    "Metadata_CellCount": pa.int32(),
}

#: Metadata columns that uniquely identify a single segmented cell.
SINGLE_CELL_KEY: tuple[str, ...] = (
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_ImageNumber",
    "Metadata_ObjectNumber",
)

#: Metadata describing a biological sample / well-level treatment. These travel
#: with every single cell and are the columns consensus profiles aggregate over.
SAMPLE_METADATA: tuple[str, ...] = (
    "Metadata_PatientID",
    "Metadata_SampleID",
    "Metadata_DiseaseGroup",
    "Metadata_FailureType",
    "Metadata_Age",
    "Metadata_Sex",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Replicate",
    "Metadata_Compound",
    "Metadata_Concentration",
    "Metadata_MoA",
)


# ---------------------------------------------------------------------------
# Column-classification helpers
# ---------------------------------------------------------------------------


def is_metadata_column(name: str) -> bool:
    """Return ``True`` when a column name is metadata rather than a feature.

    Args:
        name: The column name to classify.

    Returns:
        ``True`` if the name starts with :data:`METADATA_PREFIX`.
    """
    return name.startswith(METADATA_PREFIX)


def partition_columns(names: list[str]) -> tuple[list[str], list[str]]:
    """Split column names into ``(metadata, feature)`` lists, order preserved.

    Args:
        names: Column names to partition.

    Returns:
        A two-tuple ``(metadata_columns, feature_columns)``.
    """
    metadata = [name for name in names if is_metadata_column(name)]
    features = [name for name in names if not is_metadata_column(name)]
    return metadata, features


def feature_columns(names: list[str]) -> list[str]:
    """Return only the morphology-feature column names."""
    return partition_columns(names)[1]


def metadata_columns(names: list[str]) -> list[str]:
    """Return only the metadata column names."""
    return partition_columns(names)[0]


# ---------------------------------------------------------------------------
# Feature-name construction (CellProfiler-style)
# ---------------------------------------------------------------------------


def feature_name(
    compartment: str,
    family: str,
    measurement: str,
    channel: str | None = None,
) -> str:
    """Build a CellProfiler-style feature column name.

    The resulting names follow the ``Compartment_Family_Measurement[_Channel]``
    pattern used by CellProfiler and consumed by pycytominer, e.g.
    ``Cells_Intensity_MeanIntensity_DNA`` or ``Nuclei_AreaShape_Area``.

    Args:
        compartment: One of :data:`COMPARTMENTS` (e.g. ``"Cells"``).
        family: Feature family, e.g. ``"AreaShape"`` or ``"Intensity"``.
        measurement: The specific measurement, e.g. ``"MeanIntensity"``.
        channel: Optional imaging channel from :data:`CHANNELS`.

    Returns:
        The assembled feature-column name.
    """
    parts = [compartment, family, measurement]
    if channel is not None:
        parts.append(channel)
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------


def _metadata_fields(names: tuple[str, ...]) -> list[pa.Field]:
    """Build Arrow fields for the named canonical metadata columns."""
    missing = [name for name in names if name not in CANONICAL_METADATA_TYPES]
    if missing:
        raise KeyError(f"Unknown canonical metadata columns: {sorted(missing)}")
    return [pa.field(name, CANONICAL_METADATA_TYPES[name]) for name in names]


def single_cell_schema(
    feature_names: list[str],
    metadata_fields: tuple[str, ...] = SINGLE_CELL_KEY + SAMPLE_METADATA,
) -> pa.Schema:
    """Build the canonical single-cell table schema.

    A single-cell table has one row per segmented object: canonical metadata
    columns followed by ``float64`` morphology-feature columns.

    Args:
        feature_names: Morphology-feature column names (must not be prefixed
            with :data:`METADATA_PREFIX`).
        metadata_fields: Canonical metadata columns to include, in order.
            Duplicates (e.g. ``Metadata_Plate`` appears in both the key and the
            sample metadata) are de-duplicated while preserving first position.

    Returns:
        A :class:`pyarrow.Schema` for the single-cell table.
    """
    bad = [name for name in feature_names if is_metadata_column(name)]
    if bad:
        raise ValueError(
            f"Feature names must not use the '{METADATA_PREFIX}' prefix: {bad}"
        )

    ordered_metadata = list(dict.fromkeys(metadata_fields))
    fields = _metadata_fields(tuple(ordered_metadata))
    fields += [pa.field(name, pa.float64()) for name in feature_names]
    return pa.schema(fields)


def profile_schema(
    feature_names: list[str],
    metadata_fields: tuple[str, ...] = (
        "Metadata_PatientID",
        "Metadata_SampleID",
        "Metadata_DiseaseGroup",
        "Metadata_FailureType",
        "Metadata_Age",
        "Metadata_Sex",
        "Metadata_Compound",
        "Metadata_MoA",
        "Metadata_CellCount",
    ),
) -> pa.Schema:
    """Build the schema for an aggregated / consensus morphology profile.

    Profiles carry patient- and sample-level metadata plus aggregated feature
    values (one row per aggregation group rather than per cell).

    Args:
        feature_names: Aggregated morphology-feature column names.
        metadata_fields: Canonical metadata columns to retain on the profile.

    Returns:
        A :class:`pyarrow.Schema` for the profile table.
    """
    ordered_metadata = list(dict.fromkeys(metadata_fields))
    fields = _metadata_fields(tuple(ordered_metadata))
    fields += [pa.field(name, pa.float64()) for name in feature_names]
    return pa.schema(fields)


def image_schema() -> pa.Schema:
    """Schema for the site-level BBBC021 image/treatment table.

    One row per imaging site (field of view), carrying the real BBBC021
    plate/well/treatment metadata used to anchor the synthetic single cells.
    This is the canonical analog of a CellProfiler ``Image`` table.
    """
    return pa.schema(
        _metadata_fields(
            (
                "Metadata_TableNumber",
                "Metadata_ImageNumber",
                "Metadata_Plate",
                "Metadata_Well",
                "Metadata_Site",
                "Metadata_Replicate",
                "Metadata_Compound",
                "Metadata_Concentration",
                "Metadata_MoA",
            )
        )
    )


def patient_schema() -> pa.Schema:
    """Schema for ``patient.parquet`` (one row per synthetic patient)."""
    return pa.schema(
        _metadata_fields(
            (
                "Metadata_PatientID",
                "Metadata_DiseaseGroup",
                "Metadata_FailureType",
                "Metadata_Age",
                "Metadata_Sex",
            )
        )
    )


def clinical_schema() -> pa.Schema:
    """Schema for ``clinical.parquet`` (one row per patient).

    A deliberately small mock clinical table: enough columns to demonstrate
    schema design and joins, not a real clinical data model.
    """
    return pa.schema(
        [
            pa.field("Metadata_PatientID", pa.string()),
            pa.field("ejection_fraction", pa.float64()),
            pa.field("nyha_class", pa.int32()),
            pa.field("bnp_pg_per_ml", pa.float64()),
            pa.field("on_beta_blocker", pa.bool_()),
            pa.field("days_since_diagnosis", pa.int32()),
        ]
    )


def snrna_summary_schema() -> pa.Schema:
    """Schema for ``snrna_summary.parquet`` (per patient x cell type).

    snRNA-seq support is intentionally limited to schema integration and mock
    summary tables (no single-cell expression matrices). One row summarizes a
    cell type for a patient.
    """
    return pa.schema(
        [
            pa.field("Metadata_PatientID", pa.string()),
            pa.field("cell_type", pa.string()),
            pa.field("n_nuclei", pa.int32()),
            pa.field("fraction_of_sample", pa.float64()),
            pa.field("mean_genes_per_nucleus", pa.float64()),
            pa.field("marker_score", pa.float64()),
        ]
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class SchemaValidationError(ValueError):
    """Raised when a table does not conform to an expected schema."""


def validate_schema(
    table: pa.Table,
    expected: pa.Schema,
    *,
    allow_extra_columns: bool = False,
) -> list[str]:
    """Validate a table against an expected schema and return any problems.

    Args:
        table: The Arrow table to validate.
        expected: The schema the table should conform to.
        allow_extra_columns: When ``True``, columns present in ``table`` but not
            in ``expected`` are permitted (useful for feature columns layered on
            top of a required metadata core).

    Returns:
        A list of human-readable problem descriptions. An empty list means the
        table conforms.
    """
    problems: list[str] = []
    actual_types = dict(zip(table.schema.names, table.schema.types))

    for field in expected:
        if field.name not in actual_types:
            problems.append(f"missing required column '{field.name}'")
        elif not field.type.equals(actual_types[field.name]):
            problems.append(
                f"column '{field.name}' has type {actual_types[field.name]}, "
                f"expected {field.type}"
            )

    if not allow_extra_columns:
        extra = [name for name in table.schema.names if name not in expected.names]
        if extra:
            problems.append(f"unexpected columns: {sorted(extra)}")

    return problems


def require_schema(
    table: pa.Table,
    expected: pa.Schema,
    *,
    allow_extra_columns: bool = False,
) -> None:
    """Validate a table and raise :class:`SchemaValidationError` on any problem.

    Args:
        table: The Arrow table to validate.
        expected: The schema the table should conform to.
        allow_extra_columns: Forwarded to :func:`validate_schema`.

    Raises:
        SchemaValidationError: If the table does not conform.
    """
    problems = validate_schema(table, expected, allow_extra_columns=allow_extra_columns)
    if problems:
        raise SchemaValidationError(
            "table does not conform to expected schema: " + "; ".join(problems)
        )


def cast_to_schema(
    table: pa.Table,
    expected: pa.Schema,
    *,
    allow_extra_columns: bool = False,
) -> pa.Table:
    """Coerce a table to an expected schema, preserving column order.

    This smooths over frontend interop quirks (for example, Polars emits
    ``large_string`` while the canonical schema uses ``string``, and integer
    widths often differ) by explicitly casting each canonical column to its
    declared Arrow type.

    Args:
        table: The Arrow table to coerce (e.g. from ``polars.DataFrame.to_arrow``
            or ``pyarrow.Table.from_pandas``).
        expected: The target schema.
        allow_extra_columns: When ``True``, columns not in ``expected`` (such as
            dynamically named feature columns) are appended after the canonical
            columns unchanged.

    Returns:
        A new table whose canonical columns match ``expected`` exactly.

    Raises:
        SchemaValidationError: If a required column is missing.
    """
    missing = [name for name in expected.names if name not in table.column_names]
    if missing:
        raise SchemaValidationError(f"cannot cast; missing columns: {sorted(missing)}")

    casted = table.select(list(expected.names)).cast(expected)
    if allow_extra_columns:
        for name in table.column_names:
            if name not in expected.names:
                casted = casted.append_column(name, table.column(name))
    return casted


def write_parquet(
    table: pa.Table,
    path: str | Path,
    *,
    schema: pa.Schema | None = None,
    allow_extra_columns: bool = False,
) -> Path:
    """Write an Arrow table to Parquet, optionally enforcing a schema.

    Parent directories are created as needed. When ``schema`` is provided the
    table is cast to it first (see :func:`cast_to_schema`), guaranteeing that
    on-disk Parquet always matches the canonical data model.

    Args:
        table: The Arrow table to persist.
        path: Destination ``.parquet`` path.
        schema: Optional canonical schema to enforce.
        allow_extra_columns: Forwarded to :func:`cast_to_schema`.

    Returns:
        The resolved destination path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if schema is not None:
        table = cast_to_schema(table, schema, allow_extra_columns=allow_extra_columns)
    pq.write_table(table, dest)
    return dest


def read_parquet(path: str | Path) -> pa.Table:
    """Read a Parquet file into an Arrow table."""
    return pq.read_table(Path(path))
