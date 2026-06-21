"""
pycytominer profiling workflow: normalization, feature selection, aggregation,
and consensus profiling.

This module turns the canonical single-cell table into patient-aware morphology
profiles using the standard cytomining recipe, delegating every analytical step
to pycytominer (Milestone 5):

    single cells
      -> aggregate (per well/sample, median)
      -> normalize (robust/standardized features)
      -> feature_select (variance + correlation + NA pruning)
      -> consensus (per patient, median across replicate wells)

The final per-patient consensus is the project's "patient morphology profile".
pycytominer is imported lazily to keep ``import mock_patient_profile`` light.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import schema
from .paths import DataPaths, get_data_paths

#: Metadata columns defining a well-level sample for aggregation.
WELL_STRATA: tuple[str, ...] = (
    "Metadata_PatientID",
    "Metadata_SampleID",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Batch",
    "Metadata_DiseaseGroup",
    "Metadata_FailureType",
    "Metadata_Age",
    "Metadata_Sex",
    "Metadata_Compound",
    "Metadata_Concentration",
    "Metadata_MoA",
)

#: Replicate-defining columns for per-patient consensus profiles.
PATIENT_REPLICATE_COLUMNS: tuple[str, ...] = (
    "Metadata_PatientID",
    "Metadata_DiseaseGroup",
    "Metadata_FailureType",
    "Metadata_Age",
    "Metadata_Sex",
)

#: Default pycytominer feature-selection operations.
DEFAULT_FEATURE_SELECT_OPS: tuple[str, ...] = (
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
)

#: Default normalization method (z-score). Use ``"mad_robustize"`` for a robust
#: (median/MAD) normalization, or ``"spherize"`` for whitening.
DEFAULT_NORMALIZE_METHOD = "standardize"


def _features(profiles: pd.DataFrame, features: Sequence[str] | None) -> list[str]:
    """Resolve the feature column list (inferred from the frame when ``None``)."""
    return (
        list(features)
        if features is not None
        else schema.feature_columns(list(profiles.columns))
    )


def aggregate_to_wells(
    single_cells: pd.DataFrame,
    *,
    operation: str = "median",
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate single cells to one profile per well/sample.

    Args:
        single_cells: Canonical single-cell frame.
        operation: pycytominer aggregation operation (``median``/``mean``).
        features: Feature columns (inferred when ``None``).

    Returns:
        A per-well profile frame including a ``Metadata_Object_Count`` cell count.
    """
    import pycytominer

    strata = [col for col in WELL_STRATA if col in single_cells.columns]
    return pycytominer.aggregate(
        single_cells,
        strata=strata,
        features=_features(single_cells, features),
        operation=operation,
        compute_object_count=True,
    )


def normalize_profiles(
    profiles: pd.DataFrame,
    *,
    method: str = DEFAULT_NORMALIZE_METHOD,
    samples: str = "all",
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Normalize profile features with pycytominer.

    Args:
        profiles: Profiles to normalize (e.g. well-level).
        method: ``standardize`` (z-score), ``mad_robustize`` (robust), or
            ``spherize``.
        samples: Rows used to fit the normalization (``"all"`` or a pandas query
            such as ``"Metadata_MoA == 'DMSO'"`` for control-based normalization).
        features: Feature columns (inferred when ``None``).

    Returns:
        The normalized profile frame.
    """
    import pycytominer

    meta_features = schema.metadata_columns(list(profiles.columns))
    return pycytominer.normalize(
        profiles,
        features=_features(profiles, features),
        meta_features=meta_features,
        method=method,
        samples=samples,
    )


def select_features(
    profiles: pd.DataFrame,
    *,
    operations: Sequence[str] = DEFAULT_FEATURE_SELECT_OPS,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Reduce features with pycytominer feature selection.

    Args:
        profiles: Profiles to trim.
        operations: pycytominer feature-selection operations.
        features: Feature columns to consider (inferred when ``None``).

    Returns:
        The profile frame with low-value features removed.
    """
    import pycytominer

    return pycytominer.feature_select(
        profiles,
        features=_features(profiles, features),
        operation=list(operations),
    )


def consensus_profiles(
    profiles: pd.DataFrame,
    *,
    replicate_columns: Sequence[str] = PATIENT_REPLICATE_COLUMNS,
    operation: str = "median",
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Collapse replicate profiles into one consensus profile per group.

    Args:
        profiles: Profiles to collapse (e.g. well-level).
        replicate_columns: Grouping columns (default: per patient).
        operation: ``median``/``mean``/``modz``.
        features: Feature columns (inferred when ``None``).

    Returns:
        One consensus profile row per unique ``replicate_columns`` combination.
    """
    import pycytominer

    columns = [col for col in replicate_columns if col in profiles.columns]
    return pycytominer.consensus(
        profiles,
        replicate_columns=columns,
        operation=operation,
        features=_features(profiles, features),
    )


def build_patient_profiles(
    single_cells: pd.DataFrame,
    paths: DataPaths | None = None,
    *,
    method: str = DEFAULT_NORMALIZE_METHOD,
    normalize_samples: str = "all",
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full profiling pipeline and return (and optionally persist) it.

    Args:
        single_cells: Canonical single-cell frame.
        paths: Data locations. Defaults to :func:`get_data_paths`.
        method: Normalization method.
        normalize_samples: Rows used to fit normalization.
        write: When ``True``, write ``well_profiles.parquet`` and
            ``morphology_profile.parquet`` under ``processed``.

    Returns:
        Dict with ``well_profiles``, ``normalized``, ``selected``, and
        ``patient_profiles`` (the per-patient consensus morphology profile).
    """
    well_profiles = aggregate_to_wells(single_cells)
    normalized = normalize_profiles(
        well_profiles, method=method, samples=normalize_samples
    )
    selected = select_features(normalized)
    patient_profiles = consensus_profiles(selected)

    if write:
        paths = (paths or get_data_paths()).ensure()
        well_profiles.to_parquet(paths.processed / "well_profiles.parquet", index=False)
        patient_profiles.to_parquet(
            paths.processed / "morphology_profile.parquet", index=False
        )

    return {
        "well_profiles": well_profiles,
        "normalized": normalized,
        "selected": selected,
        "patient_profiles": patient_profiles,
    }
