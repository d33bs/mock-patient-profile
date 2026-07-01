"""
Mock multi-omic tables and patient-level integration (Milestone 8).

Generates two small, schema-conformant mock tables alongside the patient and
morphology profile tables, then demonstrates a patient-level integration join.
Per the milestone, the emphasis is on *schema design and joins*, not real
single-cell genomics: snRNA-seq support is limited to a summary table (no
expression matrices).

The integration join is performed with DuckDB directly over the Parquet files,
showcasing the project's Parquet-first, DuckDB-friendly storage model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import schema
from .paths import DataPaths, get_data_paths
from .profiling import PATIENT_REPLICATE_COLUMNS

#: snRNA-seq cell types summarized per patient (fibroblast-screen relevant).
DEFAULT_CELL_TYPES: tuple[str, ...] = (
    "Fibroblast",
    "Cardiomyocyte",
    "Endothelial",
    "Immune",
)

#: Per-disease clinical baselines: (ejection_fraction, nyha_class, bnp, bb_prob).
_CLINICAL_BASELINE: dict[str, tuple[float, int, float, float]] = {
    "Healthy": (62.0, 1, 40.0, 0.05),
    "Stable SV": (52.0, 2, 180.0, 0.35),
    "Fontan Failure": (42.0, 3, 600.0, 0.70),
    "Systolic Failure": (30.0, 4, 1200.0, 0.85),
}


def build_clinical_table(
    patient_table: pd.DataFrame,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a mock clinical table (one row per patient), conforming to schema.

    Clinical values are correlated with disease group (failing groups have lower
    ejection fraction and higher NYHA class / BNP), with seeded noise.

    Args:
        patient_table: Patient cohort table (needs PatientID + DiseaseGroup).
        seed: RNG seed.

    Returns:
        A frame conforming to :func:`schema.clinical_schema`.
    """
    rng = np.random.default_rng(seed)
    diseases = patient_table["Metadata_DiseaseGroup"].to_list()
    ef, nyha, bnp, on_bb, days = [], [], [], [], []
    for disease in diseases:
        base_ef, base_nyha, base_bnp, bb_prob = _CLINICAL_BASELINE[disease]
        ef.append(round(float(base_ef + rng.normal(0, 4)), 1))
        nyha.append(int(np.clip(base_nyha + rng.integers(-1, 2), 1, 4)))
        bnp.append(round(float(max(base_bnp * rng.lognormal(0, 0.3), 1.0)), 1))
        on_bb.append(bool(rng.random() < bb_prob))
        days.append(int(rng.integers(30, 3650)))

    return pd.DataFrame(
        {
            "Metadata_PatientID": patient_table["Metadata_PatientID"].to_list(),
            "ejection_fraction": ef,
            "nyha_class": nyha,
            "bnp_pg_per_ml": bnp,
            "on_beta_blocker": on_bb,
            "days_since_diagnosis": days,
        }
    )


def build_snrna_summary_table(
    patient_table: pd.DataFrame,
    *,
    seed: int = 0,
    cell_types: tuple[str, ...] = DEFAULT_CELL_TYPES,
) -> pd.DataFrame:
    """Build a mock snRNA-seq summary table (one row per patient x cell type).

    Args:
        patient_table: Patient cohort table (needs PatientID).
        seed: RNG seed.
        cell_types: Cell types to summarize.

    Returns:
        A frame conforming to :func:`schema.snrna_summary_schema`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for patient in patient_table["Metadata_PatientID"].to_list():
        # Dirichlet composition, fibroblast-biased (these are fibroblast samples).
        fractions = rng.dirichlet([6.0, 2.0, 1.5, 1.0])
        total = int(rng.integers(1500, 6000))
        for cell_type, fraction in zip(cell_types, fractions):
            rows.append(
                {
                    "Metadata_PatientID": patient,
                    "cell_type": cell_type,
                    "n_nuclei": round(float(total * fraction)),
                    "fraction_of_sample": round(float(fraction), 4),
                    "mean_genes_per_nucleus": round(float(rng.normal(2500, 300)), 1),
                    "marker_score": round(float(rng.normal(0.5, 0.15)), 4),
                }
            )
    return pd.DataFrame(rows)


def write_clinical_table(
    clinical: pd.DataFrame,
    paths: DataPaths | None = None,
) -> Path:
    """Write ``clinical.parquet`` under ``processed`` (schema-enforced)."""
    paths = paths or get_data_paths()
    import pyarrow as pa

    return schema.write_parquet(
        pa.Table.from_pandas(clinical, preserve_index=False),
        paths.processed / "clinical.parquet",
        schema=schema.clinical_schema(),
    )


def write_snrna_summary_table(
    snrna: pd.DataFrame,
    paths: DataPaths | None = None,
) -> Path:
    """Write ``snrna_summary.parquet`` under ``processed`` (schema-enforced)."""
    paths = paths or get_data_paths()
    import pyarrow as pa

    return schema.write_parquet(
        pa.Table.from_pandas(snrna, preserve_index=False),
        paths.processed / "snrna_summary.parquet",
        schema=schema.snrna_summary_schema(),
    )


def build_multiomic_tables(
    patient_table: pd.DataFrame,
    paths: DataPaths | None = None,
    *,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Build and persist the mock clinical and snRNA-seq summary tables.

    Args:
        patient_table: Patient cohort table.
        paths: Data locations. Defaults to :func:`get_data_paths`.
        seed: RNG seed.

    Returns:
        Dict with ``clinical`` and ``snrna_summary`` frames.
    """
    paths = (paths or get_data_paths()).ensure()
    clinical = build_clinical_table(patient_table, seed=seed)
    snrna = build_snrna_summary_table(patient_table, seed=seed)
    write_clinical_table(clinical, paths)
    write_snrna_summary_table(snrna, paths)
    return {"clinical": clinical, "snrna_summary": snrna}


def integrate_multiomics(
    paths: DataPaths | None = None,
    *,
    write: bool = True,
) -> pl.DataFrame:
    """Join patient, clinical, snRNA-seq, and morphology tables with DuckDB.

    Produces one integrated row per patient: patient metadata + clinical fields +
    a per-patient snRNA-seq summary (total nuclei and fibroblast fraction) +
    consensus morphology features. Reads the Parquet tables directly with DuckDB.

    Args:
        paths: Data locations. Defaults to :func:`get_data_paths`.
        write: When ``True``, write ``integrated_patient.parquet`` under
            ``processed``.

    Returns:
        The integrated per-patient table as a Polars frame.
    """
    import duckdb

    paths = paths or get_data_paths()
    patient = paths.processed / "patient.parquet"
    clinical = paths.processed / "clinical.parquet"
    snrna = paths.processed / "snrna_summary.parquet"
    morphology = paths.processed / "morphology_profile.parquet"

    # Exclude morphology metadata that would clash with the patient columns,
    # keeping only the morphology feature columns from the profile table.
    exclude = ", ".join(PATIENT_REPLICATE_COLUMNS)
    query = f"""
        SELECT
            p.*,
            c.ejection_fraction,
            c.nyha_class,
            c.bnp_pg_per_ml,
            c.on_beta_blocker,
            c.days_since_diagnosis,
            s.total_nuclei,
            s.fibroblast_fraction,
            m.* EXCLUDE ({exclude})
        FROM read_parquet('{patient}') AS p
        LEFT JOIN read_parquet('{clinical}') AS c USING (Metadata_PatientID)
        LEFT JOIN (
            SELECT
                Metadata_PatientID,
                SUM(n_nuclei) AS total_nuclei,
                SUM(
                    CASE WHEN cell_type = 'Fibroblast'
                    THEN fraction_of_sample ELSE 0 END
                ) AS fibroblast_fraction
            FROM read_parquet('{snrna}')
            GROUP BY Metadata_PatientID
        ) AS s USING (Metadata_PatientID)
        LEFT JOIN read_parquet('{morphology}') AS m USING (Metadata_PatientID)
        ORDER BY p.Metadata_PatientID
    """
    with duckdb.connect() as con:
        integrated = con.execute(query).pl()

    if write:
        paths.ensure()
        schema.write_parquet(
            integrated.to_arrow(), paths.processed / "integrated_patient.parquet"
        )
    return integrated
