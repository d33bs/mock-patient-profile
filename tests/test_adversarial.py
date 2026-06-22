"""
Tie-in test: the generator's difficulty knobs move the evaluation metrics.

Demonstrates the analyst feedback loop -- cranking the batch signal in
:mod:`mock_patient_profile.synthetic` makes the batch-leakage metrics in
:mod:`mock_patient_profile.evaluation` measurably worse. Uses the lightweight
stack only (numpy/polars + scikit-learn); no CytoTable/pycytominer.
"""

import polars as pl

from mock_patient_profile import bbbc021, evaluation, patients, synthetic


def _subset(n_wells: int = 10) -> pl.DataFrame:
    rows = []
    image_number = 1
    for plate in ("PlateA", "PlateB"):
        for well_num in range(1, n_wells + 1):
            rows.append(
                {
                    "Metadata_TableNumber": str(image_number),
                    "Metadata_ImageNumber": image_number,
                    "Metadata_Plate": plate,
                    "Metadata_Well": f"A{well_num:02d}",
                    "Metadata_Site": 1,
                    "Metadata_Replicate": 1,
                    "Metadata_Compound": "DMSO",
                    "Metadata_Concentration": 0.0,
                    "Metadata_MoA": "DMSO",
                }
            )
            image_number += 1
    return pl.DataFrame(rows).select(bbbc021.IMAGE_COLUMNS)


def _well_profiles(*, batch_weight: float, seed: int = 0) -> pl.DataFrame:
    """Build well-level profiles (mean over cells) for a given batch strength."""
    augmented = patients.assign_patients(_subset(), n_patients=8, seed=seed)
    cells = synthetic.simulate_single_cells(
        augmented,
        cells_per_site=20,
        seed=seed,
        signal=synthetic.SignalConfig(weight_plate=batch_weight),
    ).to_pandas()
    features = synthetic.canonical_feature_names()
    metadata = ["Metadata_Batch", "Metadata_DiseaseGroup", "Metadata_PatientID"]
    aggregated = cells.groupby("Metadata_SampleID")[features].mean().reset_index()
    carried = cells.groupby("Metadata_SampleID")[metadata].first().reset_index()
    return aggregated.merge(carried, on="Metadata_SampleID")


def _metric(card: pl.DataFrame, name: str) -> float:
    return float(card.loc[card["metric"] == name, "value"].iloc[0])


def test_batch_weight_worsens_batch_leakage_metrics() -> None:
    easy = evaluation.benchmark(_well_profiles(batch_weight=0.1))
    hard = evaluation.benchmark(_well_profiles(batch_weight=4.0))

    # stronger batch signal => more batch leakage (higher classifier accuracy)
    assert _metric(hard, "batch_classifier_balacc") > _metric(
        easy, "batch_classifier_balacc"
    )
    # and worse batch mixing (ratio further below 1.0)
    assert _metric(hard, "batch_mixing_ratio") < _metric(easy, "batch_mixing_ratio")
