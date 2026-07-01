"""
CytoDataFrame integration layer.

CytoDataFrame is a pandas ``DataFrame`` subclass from the Way Science ecosystem
that adds single-cell-image-aware display and metadata handling. This module
loads the canonical single-cell Parquet into a CytoDataFrame and provides small,
patient-aware helpers that operate over the Patient -> Sample -> Plate -> Well ->
Cell hierarchy: selection by disease group or patient, joining the patient
table, partitioning metadata vs. features, and counting cells at each level.

CytoDataFrame pulls a heavy visualization stack (pyvista/vtk), so it is imported
lazily to keep ``import mock_patient_profile`` lightweight.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import schema
from .paths import DataPaths, get_data_paths

if TYPE_CHECKING:
    from cytodataframe import CytoDataFrame

#: Map a hierarchy "level" name to the metadata column that identifies it.
LEVEL_COLUMN: dict[str, str] = {
    "patient": "Metadata_PatientID",
    "sample": "Metadata_SampleID",
    "disease": "Metadata_DiseaseGroup",
    "plate": "Metadata_Plate",
    "well": "Metadata_Well",
}


def _as_cytodataframe(data: pd.DataFrame) -> CytoDataFrame:
    """Wrap a pandas frame in a CytoDataFrame (lazy import of the heavy stack)."""
    from cytodataframe import CytoDataFrame

    return CytoDataFrame(data=data.reset_index(drop=True))


def load_single_cells(
    path: str | Path | None = None,
    paths: DataPaths | None = None,
) -> CytoDataFrame:
    """Load the canonical single-cell Parquet into a CytoDataFrame.

    Args:
        path: Explicit Parquet path. When ``None``, uses
            ``processed/single_cell.parquet`` under ``paths``.
        paths: Data locations. Defaults to :func:`get_data_paths`.

    Returns:
        A CytoDataFrame of single cells.
    """
    if path is None:
        path = (paths or get_data_paths()).processed / "single_cell.parquet"
    return _as_cytodataframe(schema.read_parquet(path).to_pandas())


def attach_patient_metadata(
    frame: pd.DataFrame,
    patient_table: pd.DataFrame | str | Path,
) -> CytoDataFrame:
    """Left-join patient attributes onto a frame by ``Metadata_PatientID``.

    Only patient columns missing from ``frame`` are added, so this is safe to
    call on a table that already carries some patient metadata (e.g. profiles).

    Args:
        frame: Any frame with a ``Metadata_PatientID`` column.
        patient_table: A patient frame or a path to ``patient.parquet``.

    Returns:
        The joined frame as a CytoDataFrame.
    """
    if isinstance(patient_table, (str, Path)):
        patient_table = schema.read_parquet(patient_table).to_pandas()

    new_columns = [
        col
        for col in patient_table.columns
        if col == "Metadata_PatientID" or col not in frame.columns
    ]
    merged = frame.merge(
        patient_table[new_columns], on="Metadata_PatientID", how="left"
    )
    return _as_cytodataframe(merged)


def select_disease_groups(
    frame: pd.DataFrame,
    groups: str | Iterable[str],
) -> CytoDataFrame:
    """Return the subset of rows in the given disease group(s)."""
    wanted = [groups] if isinstance(groups, str) else list(groups)
    return _as_cytodataframe(frame[frame["Metadata_DiseaseGroup"].isin(wanted)])


def select_patients(
    frame: pd.DataFrame,
    patient_ids: str | Iterable[str],
) -> CytoDataFrame:
    """Return the subset of rows for the given patient ID(s)."""
    wanted = [patient_ids] if isinstance(patient_ids, str) else list(patient_ids)
    return _as_cytodataframe(frame[frame["Metadata_PatientID"].isin(wanted)])


def metadata_frame(frame: pd.DataFrame) -> CytoDataFrame:
    """Return only the metadata columns of a frame."""
    metadata = schema.metadata_columns(list(frame.columns))
    return _as_cytodataframe(frame[metadata])


def feature_frame(frame: pd.DataFrame) -> CytoDataFrame:
    """Return only the morphology-feature columns of a frame."""
    features = schema.feature_columns(list(frame.columns))
    return _as_cytodataframe(frame[features])


def cell_counts(frame: pd.DataFrame, level: str = "patient") -> pd.DataFrame:
    """Count rows (cells) grouped by a hierarchy level.

    Args:
        frame: A single-cell frame.
        level: One of :data:`LEVEL_COLUMN` (``patient``/``sample``/``disease``/
            ``plate``/``well``).

    Returns:
        A two-column frame ``[<level column>, n_cells]``.
    """
    if level not in LEVEL_COLUMN:
        raise ValueError(f"unknown level '{level}'; choose from {sorted(LEVEL_COLUMN)}")
    column = LEVEL_COLUMN[level]
    return (
        frame.groupby(column, observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(column, ignore_index=True)
    )


def hierarchy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the Patient -> Sample -> Plate -> Well -> Cell hierarchy sizes.

    Args:
        frame: A single-cell frame.

    Returns:
        A one-row frame with the count of distinct entities at each level.
    """
    levels: Sequence[tuple[str, str]] = (
        ("n_patients", "Metadata_PatientID"),
        ("n_samples", "Metadata_SampleID"),
        ("n_plates", "Metadata_Plate"),
    )
    summary = {name: int(frame[column].nunique()) for name, column in levels}
    # a "well" is a distinct (plate, well) location, since well labels repeat
    summary["n_wells"] = int(
        frame[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
    )
    summary["n_cells"] = len(frame)
    return pd.DataFrame([summary])
