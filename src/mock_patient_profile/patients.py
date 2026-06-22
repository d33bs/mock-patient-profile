"""
Synthetic patient metadata layer.

This module fabricates a small, deterministic patient cohort and maps the real
BBBC021 wells onto patients, so the rest of the workflow can be exercised in a
*patient-aware* way before any real patient-derived fibroblast data exists.

The cohort mirrors a single-ventricle (SV) congenital heart disease study: four
disease groups (:data:`mock_patient_profile.schema.DISEASE_GROUPS`) each with an
associated failure phenotype, age range, and sex. Every BBBC021 well is treated
as one biological *sample* drawn from a patient; a patient contributes several
wells spanning plates and treatments, which yields realistic replicate and batch
structure for the downstream QC and normalization steps.

Nothing here is medically meaningful -- it exists purely to shape the mock so
disease-group structure, treatment response, and plate/batch effects are all
present and separable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import schema
from .paths import DataPaths, get_data_paths

#: Default number of synthetic patients in the cohort.
DEFAULT_N_PATIENTS = 8

#: Default RNG seed for reproducible patient generation.
DEFAULT_SEED = 0

#: Failure phenotype implied by each disease group.
FAILURE_TYPE_BY_DISEASE: dict[str, str] = {
    "Healthy": "None",
    "Stable SV": "Compensated",
    "Fontan Failure": "Fontan",
    "Systolic Failure": "Systolic",
}

#: Plausible age range (years, inclusive low / exclusive high) per disease group.
_AGE_RANGE_BY_DISEASE: dict[str, tuple[int, int]] = {
    "Healthy": (8, 41),
    "Stable SV": (2, 19),
    "Fontan Failure": (10, 36),
    "Systolic Failure": (15, 46),
}


def build_patient_table(
    n_patients: int = DEFAULT_N_PATIENTS,
    *,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Build the synthetic patient cohort table (one row per patient).

    Disease groups are assigned round-robin for balance; age and sex are drawn
    from a seeded RNG so the cohort is fully reproducible.

    Args:
        n_patients: Number of patients to generate.
        seed: RNG seed.

    Returns:
        A Polars frame conforming to :func:`schema.patient_schema`.
    """
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1")

    rng = np.random.default_rng(seed)
    groups = schema.DISEASE_GROUPS
    patient_ids = [f"P{idx:03d}" for idx in range(1, n_patients + 1)]
    disease = [groups[idx % len(groups)] for idx in range(n_patients)]
    failure = [FAILURE_TYPE_BY_DISEASE[group] for group in disease]
    ages = [int(rng.integers(*_AGE_RANGE_BY_DISEASE[group])) for group in disease]
    sexes = [schema.SEXES[int(rng.integers(0, len(schema.SEXES)))] for _ in disease]

    return pl.DataFrame(
        {
            "Metadata_PatientID": patient_ids,
            "Metadata_DiseaseGroup": disease,
            "Metadata_FailureType": failure,
            "Metadata_Age": pl.Series(ages, dtype=pl.Int32),
            "Metadata_Sex": sexes,
        }
    )


def _confounded_patient_indices(
    plates: list[str],
    n_patients: int,
    diseases: list[str],
    confounding: float,
    seed: int,
) -> list[int]:
    """Assign each (sorted) well to a patient index, optionally confounding plate.

    At ``confounding=0`` this is the plain global round-robin (each well's patient
    is its global index mod ``n_patients``). At ``confounding=1`` every well is
    assigned to a patient whose disease's "home plate" matches the well's plate,
    so disease group becomes predictable from plate (a realistic batch confound).
    Intermediate values mix the two per well via a seeded draw.
    """
    unique_plates = sorted(set(plates))
    n_plates = len(unique_plates)
    plate_index = {plate: i for i, plate in enumerate(unique_plates)}
    disease_index = {group: i for i, group in enumerate(schema.DISEASE_GROUPS)}

    # Each disease group gets a home plate; each plate's "home patients" are those
    # whose disease maps there (fall back to all patients if a plate has none).
    home_plate = [disease_index[diseases[p]] % n_plates for p in range(n_patients)]
    home_patients = {
        pi: [p for p in range(n_patients) if home_plate[p] == pi]
        for pi in range(n_plates)
    }
    for pi, members in home_patients.items():
        if not members:
            home_patients[pi] = list(range(n_patients))

    rng = np.random.default_rng(seed)
    plate_counter = dict.fromkeys(range(n_plates), 0)
    assignments = []
    for global_counter, plate in enumerate(plates):
        pi = plate_index[plate]
        global_idx = global_counter % n_patients
        members = home_patients[pi]
        confounded_idx = members[plate_counter[pi] % len(members)]
        plate_counter[pi] += 1
        use_confounded = rng.random() < confounding
        assignments.append(confounded_idx if use_confounded else global_idx)
    return assignments


def assign_patients(
    image_table: pl.DataFrame,
    n_patients: int = DEFAULT_N_PATIENTS,
    *,
    seed: int = DEFAULT_SEED,
    disease_plate_confounding: float = 0.0,
) -> pl.DataFrame:
    """Attach patient/sample metadata to a site-level image table.

    Each unique ``(Metadata_Plate, Metadata_Well)`` becomes a sample assigned to
    a patient. The plate doubles as the imaging batch.

    Args:
        image_table: A canonicalized site-level image table (see
            :func:`mock_patient_profile.bbbc021.build_dev_subset`).
        n_patients: Number of patients in the cohort.
        seed: RNG seed for the cohort.
        disease_plate_confounding: In ``[0, 1]``. ``0`` spreads each patient (and
            disease group) evenly across plates (the easy, balanced default).
            ``1`` concentrates each disease group on plate(s), so disease becomes
            confounded with batch -- the realistic, hard case for downstream
            normalization / batch correction to disentangle.

    Returns:
        ``image_table`` augmented with ``Metadata_SampleID``,
        ``Metadata_Batch``, ``Metadata_PatientID``, ``Metadata_DiseaseGroup``,
        ``Metadata_FailureType``, ``Metadata_Age``, and ``Metadata_Sex``.
    """
    if not 0.0 <= disease_plate_confounding <= 1.0:
        raise ValueError("disease_plate_confounding must be in [0, 1]")

    patients = build_patient_table(n_patients, seed=seed)
    diseases = patients["Metadata_DiseaseGroup"].to_list()

    wells = (
        image_table.select(["Metadata_Plate", "Metadata_Well"])
        .unique()
        .sort(["Metadata_Plate", "Metadata_Well"])
    )
    assignments = _confounded_patient_indices(
        wells["Metadata_Plate"].to_list(),
        n_patients,
        diseases,
        disease_plate_confounding,
        seed,
    )
    wells = (
        wells.with_columns(
            _patient_idx=pl.Series(assignments, dtype=pl.UInt32),
            Metadata_SampleID=pl.col("Metadata_Plate") + "_" + pl.col("Metadata_Well"),
            Metadata_Batch=pl.col("Metadata_Plate"),
        )
        .join(patients.with_row_index("_patient_idx"), on="_patient_idx", how="left")
        .drop("_patient_idx")
    )

    return image_table.join(wells, on=["Metadata_Plate", "Metadata_Well"], how="left")


def write_patient_table(
    patients: pl.DataFrame,
    paths: DataPaths | None = None,
) -> Path:
    """Write the patient cohort to canonical Parquet (``processed/patient.parquet``).

    Args:
        patients: A patient table (e.g. from :func:`build_patient_table`).
        paths: Data locations. Defaults to :func:`get_data_paths`.

    Returns:
        The path of the written Parquet file.
    """
    paths = paths or get_data_paths()
    dest = paths.processed / "patient.parquet"
    return schema.write_parquet(
        patients.to_arrow(), dest, schema=schema.patient_schema()
    )
