"""
Tests for the evaluation/metrics module.

Each metric is checked against synthetic profiles with controllable disease
separation, batch effect, and patient-specific signal.
"""

import numpy as np
import pandas as pd

from mock_patient_profile import evaluation

GROUPS = ["Healthy", "Stable SV", "Fontan Failure", "Systolic Failure"]


def _profiles(
    *,
    disease_sep: float = 3.0,
    batch_sep: float = 0.0,
    patient_sep: float = 0.0,
    n_per_group: int = 24,
    n_features: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    disease_centroid = {g: rng.normal(0, disease_sep, n_features) for g in GROUPS}
    rows = []
    for gi, group in enumerate(GROUPS):
        patients = [f"PT{gi}_{p}" for p in range(4)]
        patient_offset = {p: rng.normal(0, patient_sep, n_features) for p in patients}
        for j in range(n_per_group):
            patient = patients[j % 4]
            batch = "P1" if j % 2 == 0 else "P2"
            x = (
                disease_centroid[group]
                + patient_offset[patient]
                + rng.normal(0, 1, n_features)
            )
            if batch == "P2":
                x = x + batch_sep
            row = {
                "Metadata_DiseaseGroup": group,
                "Metadata_Batch": batch,
                "Metadata_PatientID": patient,
            }
            row.update(dict(zip(features, x)))
            rows.append(row)
    return pd.DataFrame(rows)


def test_disease_metrics_detect_separation() -> None:
    profiles = _profiles(disease_sep=4.0, batch_sep=0.0)
    assert evaluation.silhouette(profiles, "Metadata_DiseaseGroup") > 0.1
    balacc = evaluation.label_classifiability(profiles, "Metadata_DiseaseGroup")
    assert balacc > evaluation.chance_level(profiles, "Metadata_DiseaseGroup") + 0.2


def test_no_batch_signal_is_well_mixed() -> None:
    profiles = _profiles(disease_sep=3.0, batch_sep=0.0)
    batch_balacc = evaluation.label_classifiability(profiles, "Metadata_Batch")
    assert batch_balacc < 0.65  # near chance (0.5)
    assert evaluation.batch_mixing(profiles, "Metadata_Batch") > 0.8


def test_batch_signal_is_detected() -> None:
    profiles = _profiles(disease_sep=3.0, batch_sep=8.0)
    batch_balacc = evaluation.label_classifiability(profiles, "Metadata_Batch")
    assert batch_balacc > 0.85
    assert evaluation.batch_mixing(profiles, "Metadata_Batch") < 0.7


def test_replicate_reproducibility_responds_to_patient_signal() -> None:
    weak = evaluation.replicate_reproducibility(_profiles(patient_sep=0.0))
    strong = evaluation.replicate_reproducibility(_profiles(patient_sep=5.0))
    assert 0.0 <= weak <= 1.0
    assert strong > weak


def test_benchmark_scorecard() -> None:
    profiles = _profiles(disease_sep=4.0, batch_sep=0.0)
    card = evaluation.benchmark(profiles)
    assert list(card.columns) == ["metric", "value", "interpretation"]
    assert set(card["metric"]) == {
        "replicate_reproducibility_map",
        "disease_silhouette",
        "disease_classifier_balacc",
        "batch_classifier_balacc",
        "batch_mixing_ratio",
    }
    values = dict(zip(card["metric"], card["value"]))
    chance = evaluation.chance_level(profiles, "Metadata_DiseaseGroup")
    assert values["disease_classifier_balacc"] > chance


def test_metrics_degrade_gracefully_on_tiny_input() -> None:
    tiny = pd.DataFrame(
        {
            "Metadata_DiseaseGroup": ["Healthy", "Fontan Failure"],
            "Metadata_Batch": ["P1", "P2"],
            "Metadata_PatientID": ["P001", "P002"],
            "Cells_AreaShape_F0": [1.0, 2.0],
        }
    )
    card = evaluation.benchmark(tiny)
    # no crash; metrics that cannot be computed are NaN
    assert card["value"].isna().any()
