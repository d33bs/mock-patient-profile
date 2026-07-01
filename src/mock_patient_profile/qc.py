"""
coSMicQC-based quality-control reporting workflow.

Produces a small, reproducible single-cell QC report covering the dimensions
called for in Milestone 4: cell counts, feature missingness, feature drift
across batches, plate effects, and single-cell outlier detection. Outlier
detection is delegated to coSMicQC (z-score threshold based); the remaining
summaries are simple, transparent Polars/pandas aggregations over the canonical
single-cell table.

The report is returned as a dictionary of tidy tables and (optionally) written
to the ``reports`` directory as Parquet plus a human-readable Markdown summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cytodataframe_io as cdf_io
from . import schema
from .paths import DataPaths, get_data_paths

#: Default coSMicQC outlier thresholds (feature -> z-score cutoff, upper tail).
#: These target biologically intuitive QC failures: oversized or unusually bright
#: objects (clumps / debris). Features absent from a table are ignored.
DEFAULT_OUTLIER_THRESHOLDS: dict[str, float] = {
    "Nuclei_AreaShape_Area": 3.0,
    "Cells_AreaShape_Area": 3.0,
    "Nuclei_Intensity_IntegratedIntensity_DNA": 3.0,
}


def cell_count_summary(single_cells: pd.DataFrame, level: str = "well") -> pd.DataFrame:
    """Count cells grouped by a hierarchy level (see :data:`cdf_io.LEVEL_COLUMN`)."""
    return cdf_io.cell_counts(single_cells, level=level)


def missingness(single_cells: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature missingness.

    Returns:
        A frame ``[feature, n_missing, fraction_missing]`` sorted worst-first.
    """
    features = schema.feature_columns(list(single_cells.columns))
    n_rows = len(single_cells)
    missing = single_cells[features].isna().sum()
    out = pd.DataFrame(
        {
            "feature": features,
            "n_missing": missing.to_numpy(),
            "fraction_missing": missing.to_numpy() / n_rows if n_rows else 0.0,
        }
    )
    return out.sort_values("fraction_missing", ascending=False, ignore_index=True)


def feature_drift(
    single_cells: pd.DataFrame,
    by: str = "Metadata_Batch",
) -> pd.DataFrame:
    """Quantify per-feature drift across groups (default: batch).

    The drift score is the standard deviation of per-group feature means divided
    by the overall feature standard deviation -- i.e. how much a feature's center
    moves between groups relative to its total spread. Larger means more drift.

    Args:
        single_cells: Canonical single-cell frame.
        by: Grouping metadata column (e.g. ``Metadata_Batch`` or
            ``Metadata_Plate``).

    Returns:
        A frame ``[feature, drift_score]`` sorted most-drifting-first.
    """
    features = schema.feature_columns(list(single_cells.columns))
    overall_std = single_cells[features].std(ddof=0).replace(0, np.nan)
    group_means = single_cells.groupby(by, observed=True)[features].mean()
    drift = (group_means.std(ddof=0) / overall_std).fillna(0.0)
    return pd.DataFrame(
        {"feature": features, "drift_score": drift.to_numpy()}
    ).sort_values("drift_score", ascending=False, ignore_index=True)


def detect_outliers(
    single_cells: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag single-cell outliers with coSMicQC.

    Each threshold feature is evaluated independently (a cell is an outlier if it
    fails any threshold), which is exposed both as a per-feature summary and as a
    combined ``cqc_is_outlier`` label.

    Args:
        single_cells: Canonical single-cell frame.
        thresholds: Feature -> z-score cutoff map. Defaults to
            :data:`DEFAULT_OUTLIER_THRESHOLDS`; features absent are skipped.

    Returns:
        ``(labeled, summary)`` where ``labeled`` is the input frame with a
        boolean ``cqc_is_outlier`` column and ``summary`` is a per-feature frame
        ``[feature, threshold, n_outliers, fraction_outliers]``.
    """
    import cosmicqc

    thresholds = thresholds or DEFAULT_OUTLIER_THRESHOLDS
    frame = single_cells.reset_index(drop=True)
    present = {
        feature: cutoff
        for feature, cutoff in thresholds.items()
        if feature in frame.columns
    }

    combined = np.zeros(len(frame), dtype=bool)
    rows = []
    for feature, cutoff in present.items():
        mask = np.asarray(
            cosmicqc.identify_outliers(
                frame,
                feature_thresholds={feature: cutoff},
                feature_thresholds_file=None,
            ),
            dtype=bool,
        )
        combined |= mask
        rows.append(
            {
                "feature": feature,
                "threshold": cutoff,
                "n_outliers": int(mask.sum()),
                "fraction_outliers": float(mask.mean()) if len(frame) else 0.0,
            }
        )

    labeled = frame.copy()
    labeled["cqc_is_outlier"] = combined
    summary = pd.DataFrame(
        rows, columns=["feature", "threshold", "n_outliers", "fraction_outliers"]
    )
    return labeled, summary


def qc_report(
    single_cells: pd.DataFrame,
    paths: DataPaths | None = None,
    *,
    thresholds: dict[str, float] | None = None,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full QC workflow and return (and optionally persist) the tables.

    Args:
        single_cells: Canonical single-cell frame.
        paths: Data locations. Defaults to :func:`get_data_paths`.
        thresholds: Outlier thresholds forwarded to :func:`detect_outliers`.
        write: When ``True``, write Parquet tables and a Markdown summary under
            the ``reports`` directory.

    Returns:
        A dict of QC tables keyed by name.
    """
    _labeled, outlier_summary = detect_outliers(single_cells, thresholds=thresholds)
    report = {
        "cell_counts_per_well": cell_count_summary(single_cells, "well"),
        "cell_counts_per_plate": cell_count_summary(single_cells, "plate"),
        "cell_counts_per_disease": cell_count_summary(single_cells, "disease"),
        "missingness": missingness(single_cells),
        "feature_drift_by_batch": feature_drift(single_cells, "Metadata_Batch"),
        "plate_effects": feature_drift(single_cells, "Metadata_Plate"),
        "outliers": outlier_summary,
    }

    if write:
        paths = (paths or get_data_paths()).ensure()
        for name, table in report.items():
            table.to_parquet(paths.reports / f"qc_{name}.parquet", index=False)
        (paths.reports / "qc_report.md").write_text(
            render_markdown(single_cells, report)
        )
    return report


def render_markdown(
    single_cells: pd.DataFrame,
    report: dict[str, pd.DataFrame],
) -> str:
    """Render a compact Markdown QC summary from the report tables."""
    n_cells = len(single_cells)
    n_outliers = int(report["outliers"]["n_outliers"].sum())
    top_drift = report["feature_drift_by_batch"].head(5)
    lines = [
        "# Single-cell QC report",
        "",
        f"- **Cells:** {n_cells}",
        f"- **Patients:** {single_cells['Metadata_PatientID'].nunique()}",
        f"- **Plates:** {single_cells['Metadata_Plate'].nunique()}",
        f"- **Outlier flags (union of thresholds):** {n_outliers} "
        f"({n_outliers / n_cells:.2%})"
        if n_cells
        else "- **Outlier flags:** 0",
        "",
        "## Outlier detection (coSMicQC)",
        "",
        report["outliers"].to_markdown(index=False),
        "",
        "## Top feature drift across batches",
        "",
        top_drift.to_markdown(index=False),
        "",
        "## Cells per disease group",
        "",
        report["cell_counts_per_disease"].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
