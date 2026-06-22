"""
Quantitative evaluation metrics for morphology profiles.

Turns "did this analysis choice help?" into numbers. Given a profile table
(typically well-level, with metadata + features), :func:`benchmark` returns a
tidy scorecard you can compute before/after a normalization or batch-correction
method and compare directly.

The metrics implemented here are transparent reference implementations (numpy +
scikit-learn) so the computation is fully inspectable:

- **replicate reproducibility** -- mean average precision (mAP) of retrieving
  same-group (e.g. same-patient) profiles by cosine similarity. Higher is
  better. This mirrors the percent-replicating / mAP idea used by the
  cytomining ``copairs`` / ``cytominer-eval`` tools, which are the natural
  production replacements once you outgrow this scaffold.
- **disease separation** -- silhouette score and a cross-validated classifier
  balanced accuracy over disease group. Higher is better (the signal to
  preserve).
- **batch leakage** -- a cross-validated classifier balanced accuracy over batch.
  Higher means more batch signal leaking into the features (worse).
- **batch mixing** -- a kNN mixing ratio over batch (1.0 = perfectly mixed,
  toward 0 = batch-separated).

All metrics degrade gracefully (return ``NaN``) when there are too few samples
or classes, so they are safe to call on tiny development data.

NOTE: these metrics are only as meaningful as the data is realistic. On the
default mock (signal planted linearly, batch balanced across disease) they will
look easy; use :mod:`mock_patient_profile.synthetic` and
:func:`mock_patient_profile.patients.assign_patients` confounding controls to
make them discriminating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .paths import DataPaths, get_data_paths

#: Default neighborhood size for kNN-based metrics.
DEFAULT_K = 15

#: Default cross-validation folds for classifier-based metrics.
DEFAULT_CV_SPLITS = 5

# Minimum-data guards so metrics return NaN rather than crashing on tiny inputs.
_MIN_CLASSES = 2
_MIN_FOLDS = 2
_MIN_SAMPLES = 2
_MIN_SAMPLES_FOR_MAP = 3


def _feature_matrix(
    profiles: pd.DataFrame,
    features: list[str] | None,
) -> np.ndarray:
    """Extract a clean float feature matrix (drop all-NaN cols, median-impute)."""
    columns = features or schema.feature_columns(list(profiles.columns))
    matrix = profiles[columns].to_numpy(dtype=float)
    keep = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, keep]
    if matrix.size and np.isnan(matrix).any():
        medians = np.nanmedian(matrix, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(matrix))
        matrix[nan_rows, nan_cols] = medians[nan_cols]
    return matrix


def replicate_reproducibility(
    profiles: pd.DataFrame,
    replicate_col: str = "Metadata_PatientID",
    features: list[str] | None = None,
) -> float:
    """Mean average precision of retrieving same-``replicate_col`` profiles.

    For each profile, rank all others by cosine similarity and score how well
    same-group profiles are retrieved (average precision); average over profiles.
    Returns ``NaN`` if no profile has both same- and different-group neighbors.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.metrics.pairwise import cosine_similarity

    matrix = _feature_matrix(profiles, features)
    labels = profiles[replicate_col].to_numpy()
    if matrix.shape[0] < _MIN_SAMPLES_FOR_MAP or matrix.shape[1] == 0:
        return float("nan")

    similarity = cosine_similarity(matrix)
    average_precisions = []
    for i in range(len(labels)):
        keep = np.arange(len(labels)) != i
        y_true = (labels[keep] == labels[i]).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        average_precisions.append(average_precision_score(y_true, similarity[i, keep]))
    return float(np.mean(average_precisions)) if average_precisions else float("nan")


def silhouette(
    profiles: pd.DataFrame,
    label_col: str = "Metadata_DiseaseGroup",
    features: list[str] | None = None,
) -> float:
    """Silhouette score of ``label_col`` clusters in feature space."""
    from sklearn.metrics import silhouette_score

    matrix = _feature_matrix(profiles, features)
    labels = profiles[label_col].to_numpy()
    n_labels = len(np.unique(labels))
    if matrix.shape[1] == 0 or n_labels < _MIN_CLASSES or len(labels) <= n_labels:
        return float("nan")
    return float(silhouette_score(matrix, labels))


def label_classifiability(
    profiles: pd.DataFrame,
    label_col: str,
    features: list[str] | None = None,
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
) -> float:
    """Cross-validated balanced accuracy of predicting ``label_col`` from features.

    Uses a kNN classifier (robust, no convergence issues). Compare against the
    chance level ``1 / n_classes`` (see :func:`chance_level`): near chance means
    the label is not encoded in the features.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    matrix = _feature_matrix(profiles, features)
    labels = profiles[label_col].to_numpy()
    _, counts = np.unique(labels, return_counts=True)
    folds = min(n_splits, int(counts.min()))
    if matrix.shape[1] == 0 or len(counts) < _MIN_CLASSES or folds < _MIN_FOLDS:
        return float("nan")

    classifier = KNeighborsClassifier(n_neighbors=min(5, int(counts.min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = cross_val_score(
        classifier, matrix, labels, cv=splitter, scoring="balanced_accuracy"
    )
    return float(scores.mean())


def chance_level(profiles: pd.DataFrame, label_col: str) -> float:
    """Balanced-accuracy chance level (``1 / n_classes``) for ``label_col``."""
    n_classes = profiles[label_col].nunique()
    return 1.0 / n_classes if n_classes else float("nan")


def batch_mixing(
    profiles: pd.DataFrame,
    batch_col: str = "Metadata_Batch",
    features: list[str] | None = None,
    *,
    k: int = DEFAULT_K,
) -> float:
    """kNN batch-mixing ratio: observed / expected cross-batch neighbor fraction.

    For each profile, measure the fraction of its ``k`` nearest neighbors in a
    different batch, and divide by the fraction expected under perfect mixing.
    ``1.0`` means batches are well mixed; values toward ``0`` mean batch
    separation. Returns ``NaN`` with fewer than two batches.
    """
    from sklearn.neighbors import NearestNeighbors

    matrix = _feature_matrix(profiles, features)
    batches = profiles[batch_col].to_numpy()
    n = len(batches)
    n_batches = len(np.unique(batches))
    if matrix.shape[1] == 0 or n_batches < _MIN_CLASSES or n <= _MIN_SAMPLES:
        return float("nan")

    n_neighbors = min(k, n - 1)
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(matrix)
    neighbor_idx = finder.kneighbors(return_distance=False)[:, 1:]  # drop self
    observed = np.mean(batches[neighbor_idx] != batches[:, None])
    # expected cross-batch fraction under random mixing
    _, counts = np.unique(batches, return_counts=True)
    expected = 1.0 - np.sum((counts / n) ** 2)
    return float(observed / expected) if expected > 0 else float("nan")


def benchmark(
    profiles: pd.DataFrame,
    *,
    disease_col: str = "Metadata_DiseaseGroup",
    batch_col: str = "Metadata_Batch",
    replicate_col: str = "Metadata_PatientID",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute a tidy scorecard of profile-quality metrics.

    Call this on profiles before and after a method (normalization, batch
    correction) and compare the ``value`` columns.

    Returns:
        A frame ``[metric, value, interpretation]``.
    """
    disease_chance = chance_level(profiles, disease_col)
    batch_chance = chance_level(profiles, batch_col)
    rows = [
        (
            "replicate_reproducibility_map",
            replicate_reproducibility(profiles, replicate_col, features),
            "higher is better (same-replicate profiles look alike)",
        ),
        (
            "disease_silhouette",
            silhouette(profiles, disease_col, features),
            "higher is better (disease groups separate)",
        ),
        (
            "disease_classifier_balacc",
            label_classifiability(profiles, disease_col, features),
            f"higher is better; chance={disease_chance:.3f}",
        ),
        (
            "batch_classifier_balacc",
            label_classifiability(profiles, batch_col, features),
            f"lower is better (batch leakage); chance={batch_chance:.3f}",
        ),
        (
            "batch_mixing_ratio",
            batch_mixing(profiles, batch_col, features),
            "1.0 = mixed, toward 0 = batch-separated",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def benchmark_from_paths(
    paths: DataPaths | None = None,
    profile_file: str = "well_profiles.parquet",
    **kwargs: object,
) -> pd.DataFrame:
    """Load a processed profile table and run :func:`benchmark` on it.

    Args:
        paths: Data locations. Defaults to :func:`get_data_paths`.
        profile_file: File under ``processed`` to evaluate (well-level profiles
            are recommended -- per-patient profiles are usually too few rows).
        **kwargs: Forwarded to :func:`benchmark`.

    Returns:
        The benchmark scorecard.
    """
    paths = paths or get_data_paths()
    profiles = schema.read_parquet(paths.processed / profile_file).to_pandas()
    return benchmark(profiles, **kwargs)  # type: ignore[arg-type]
