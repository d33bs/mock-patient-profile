"""
Synthetic CellProfiler-style single-cell feature generator.

BBBC021's real per-object measurements are large and not needed to prove out the
*computational architecture*. Instead this module fabricates small, realistic
CellProfiler-style outputs (an ``Image`` table plus ``Cells`` / ``Cytoplasm`` /
``Nuclei`` compartment tables) anchored to the real BBBC021 dev-subset metadata
and the synthetic patient assignment.

Feature values are drawn so that three separable signals are baked into the
single cells, which is exactly what the downstream QC, normalization, and
(future) batch-correction steps need to demonstrate value:

- **disease group** -- a per-(disease, feature) shift (the signal to preserve),
- **mechanism of action** -- a per-(MoA, feature) treatment-response shift,
- **plate / batch** -- a per-(plate, feature) nuisance shift to correct.

:class:`SignalConfig` controls the strength of each signal and the
within-compartment feature correlation, so the difficulty can be tuned from
"trivially separable" to "batch overwhelms biology" for method benchmarking
(the defaults reproduce the original easy mock). Everything is deterministic
given a seed. The compartment tables are written as real CellProfiler-style
CSVs so that :mod:`mock_patient_profile.cytotable_io` can perform a genuine
CytoTable conversion back into single-cell Parquet.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import schema
from .paths import DataPaths, get_data_paths

#: Default number of synthetic cells generated per imaging site.
DEFAULT_CELLS_PER_SITE = 15

#: Default RNG seed for reproducible feature generation.
DEFAULT_SEED = 0


@dataclass(frozen=True)
class SignalConfig:
    """Controls how strong each planted signal is, and feature realism.

    The defaults reproduce the original easy mock (disease dominant, modest
    batch, independent features). Treat this as your experimental knob: raise
    ``weight_plate`` above ``weight_disease`` to make batch overwhelm biology,
    and raise ``feature_correlation`` to give features a realistic within-
    compartment covariance (so dimensionality reduction / feature selection
    actually matter).

    Attributes:
        weight_disease: Weight of the per-(disease, feature) signal to preserve.
        weight_moa: Weight of the per-(MoA, feature) treatment-response signal.
        weight_plate: Weight of the per-(plate, feature) batch nuisance signal.
        weight_noise: Weight of per-cell noise.
        feature_correlation: In ``[0, 1)``. Correlation injected between the
            per-cell noise of features sharing a compartment (``0`` = the
            original independent features).
    """

    weight_disease: float = 0.9
    weight_moa: float = 0.7
    weight_plate: float = 0.4
    weight_noise: float = 0.7
    feature_correlation: float = 0.0


#: AreaShape measurements (channel-independent) emitted per compartment.
_AREASHAPE_MEASUREMENTS = (
    "AreaShape_Area",
    "AreaShape_Perimeter",
    "AreaShape_FormFactor",
)
#: Per-channel intensity measurements emitted per compartment.
_INTENSITY_MEASUREMENTS = (
    "Intensity_MeanIntensity",
    "Intensity_IntegratedIntensity",
)
#: Per-channel texture measurements emitted per compartment.
_TEXTURE_MEASUREMENTS = ("Texture_Contrast",)

#: Baseline (mean, relative-standard-deviation) per measurement family.
_MEASUREMENT_BASE: dict[str, tuple[float, float]] = {
    "AreaShape_Area": (800.0, 0.18),
    "AreaShape_Perimeter": (110.0, 0.15),
    "AreaShape_FormFactor": (0.85, 0.08),
    "Intensity_MeanIntensity": (0.45, 0.20),
    "Intensity_IntegratedIntensity": (350.0, 0.25),
    "Texture_Contrast": (12.0, 0.30),
}

#: Measurement families whose magnitude scales with object size.
_SIZE_SCALED = (
    "AreaShape_Area",
    "AreaShape_Perimeter",
    "Intensity_IntegratedIntensity",
)

#: Relative object-size scale per compartment (cells > cytoplasm > nuclei).
_COMPARTMENT_SCALE: dict[str, float] = {
    "Cells": 1.6,
    "Cytoplasm": 1.3,
    "Nuclei": 0.6,
}


def compartment_measurements() -> list[str]:
    """Return the measurement names emitted per compartment (no compartment prefix).

    These are the column names that appear in a CellProfiler compartment CSV,
    e.g. ``AreaShape_Area`` or ``Intensity_MeanIntensity_DNA``.
    """
    names = list(_AREASHAPE_MEASUREMENTS)
    for channel in schema.CHANNELS:
        names += [f"{m}_{channel}" for m in _INTENSITY_MEASUREMENTS]
        names += [f"{m}_{channel}" for m in _TEXTURE_MEASUREMENTS]
    return names


def canonical_feature_names() -> list[str]:
    """Return the full canonical feature names (``Compartment_Measurement...``)."""
    return [
        f"{compartment}_{measurement}"
        for compartment in schema.COMPARTMENTS
        for measurement in compartment_measurements()
    ]


def _measurement_base(measurement: str) -> tuple[float, float]:
    """Return the (mean, relative-sd) baseline for a measurement name."""
    for prefix, base in _MEASUREMENT_BASE.items():
        if measurement.startswith(prefix):
            return base
    raise KeyError(f"no baseline configured for measurement '{measurement}'")


def _feature_means_and_sds(
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature ``(means, relative_sds)`` arrays for canonical features."""
    means = np.empty(len(feature_names))
    rel_sds = np.empty(len(feature_names))
    for idx, name in enumerate(feature_names):
        compartment, measurement = name.split("_", 1)
        mean, rel_sd = _measurement_base(measurement)
        if any(measurement.startswith(scaled) for scaled in _SIZE_SCALED):
            mean *= _COMPARTMENT_SCALE[compartment]
        means[idx] = mean
        rel_sds[idx] = rel_sd
    return means, rel_sds


def _stable_seed(label: str) -> int:
    """Map a label to a stable 32-bit seed (order-independent across runs)."""
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def _effect_matrix(
    labels: np.ndarray,
    kind: str,
    seed: int,
    n_features: int,
) -> np.ndarray:
    """Build an ``[n_cells, n_features]`` effect matrix for a categorical signal.

    Each distinct label draws a stable coefficient vector from a seed derived
    from ``(seed, kind, label)``, so a label's effect is reproducible and
    independent of how many cells or which order they appear in.
    """
    unique, inverse = np.unique(labels, return_inverse=True)
    coefficients = np.vstack(
        [
            np.random.default_rng(
                [seed, _stable_seed(kind), _stable_seed(str(label))]
            ).standard_normal(n_features)
            for label in unique
        ]
    )
    return coefficients[inverse]


def _noise_cholesky(feature_names: list[str], correlation: float) -> np.ndarray:
    """Cholesky factor of a per-compartment equicorrelation matrix.

    Features sharing a compartment get correlation ``correlation``; features in
    different compartments stay independent. Used to give the per-cell noise a
    realistic block covariance structure.
    """
    compartments = np.array([name.split("_", 1)[0] for name in feature_names])
    same_compartment = compartments[:, None] == compartments[None, :]
    sigma = np.where(same_compartment, correlation, 0.0)
    np.fill_diagonal(sigma, 1.0)
    return np.linalg.cholesky(sigma)


def simulate_single_cells(
    augmented_table: pl.DataFrame,
    *,
    cells_per_site: int = DEFAULT_CELLS_PER_SITE,
    seed: int = DEFAULT_SEED,
    signal: SignalConfig | None = None,
) -> pl.DataFrame:
    """Simulate the canonical single-cell "ground-truth" table.

    Args:
        augmented_table: A site-level image table with patient metadata attached
            (see :func:`mock_patient_profile.patients.assign_patients`). Must
            include ``Metadata_DiseaseGroup``, ``Metadata_MoA``, and
            ``Metadata_Plate``.
        cells_per_site: Number of cells to generate per imaging site.
        seed: RNG seed.
        signal: Signal strengths and feature-correlation knobs. Defaults to
            :class:`SignalConfig` (the original easy mock).

    Returns:
        A Polars frame with one row per cell: all input metadata columns plus
        ``Metadata_ObjectNumber`` and the canonical morphology features.
    """
    if cells_per_site < 1:
        raise ValueError("cells_per_site must be >= 1")
    signal = signal or SignalConfig()

    sites = augmented_table.sort("Metadata_ImageNumber")
    objects = pl.DataFrame(
        {
            "Metadata_ObjectNumber": pl.Series(
                range(1, cells_per_site + 1), dtype=pl.Int32
            )
        }
    )
    cells = sites.join(objects, how="cross")

    feature_names = canonical_feature_names()
    n_cells, n_features = cells.height, len(feature_names)
    means, rel_sds = _feature_means_and_sds(feature_names)

    disease = _effect_matrix(
        cells["Metadata_DiseaseGroup"].to_numpy(), "disease", seed, n_features
    )
    moa = _effect_matrix(
        cells["Metadata_MoA"].fill_null("DMSO").to_numpy(), "moa", seed, n_features
    )
    plate = _effect_matrix(
        cells["Metadata_Plate"].to_numpy(), "plate", seed, n_features
    )
    noise = np.random.default_rng(seed).standard_normal((n_cells, n_features))
    if signal.feature_correlation > 0:
        noise = noise @ _noise_cholesky(feature_names, signal.feature_correlation).T

    z = (
        signal.weight_disease * disease
        + signal.weight_moa * moa
        + signal.weight_plate * plate
        + signal.weight_noise * noise
    )
    values = means * (1.0 + rel_sds * z)
    # Floor at a small fraction of the mean to keep measurements non-negative.
    values = np.maximum(values, means * 0.02)

    feature_frame = pl.from_numpy(values, schema=feature_names)
    return cells.hstack(feature_frame)


def _compartment_table(cells: pl.DataFrame, compartment: str) -> pl.DataFrame:
    """Build a CellProfiler-style compartment table from the truth frame."""
    columns: dict[str, pl.Series] = {
        "ImageNumber": cells["Metadata_ImageNumber"],
        "ObjectNumber": cells["Metadata_ObjectNumber"],
    }
    for measurement in compartment_measurements():
        columns[measurement] = cells[f"{compartment}_{measurement}"].rename(measurement)
    columns["Number_Object_Number"] = cells["Metadata_ObjectNumber"]
    if compartment == "Cytoplasm":
        columns["Parent_Cells"] = cells["Metadata_ObjectNumber"]
        columns["Parent_Nuclei"] = cells["Metadata_ObjectNumber"]
    return pl.DataFrame(columns)


def _image_table(augmented_table: pl.DataFrame, cells_per_site: int) -> pl.DataFrame:
    """Build a CellProfiler-style Image table with channel filenames + cell count."""
    sites = (
        augmented_table.select(
            ["Metadata_ImageNumber", "Metadata_Plate", "Metadata_Well", "Metadata_Site"]
        )
        .unique()
        .sort("Metadata_ImageNumber")
    )
    stem = (
        pl.col("Metadata_Plate")
        + "_"
        + pl.col("Metadata_Well")
        + "_s"
        + pl.col("Metadata_Site").cast(pl.Utf8)
    )
    return sites.select(
        pl.col("Metadata_ImageNumber").alias("ImageNumber"),
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        *[
            (stem + f"_{channel}.tif").alias(f"Image_FileName_{channel}")
            for channel in schema.CHANNELS
        ],
        pl.lit(cells_per_site, dtype=pl.Int32).alias("Count_Cells"),
    )


def write_cellprofiler_csvs(
    cells: pl.DataFrame,
    augmented_table: pl.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write CellProfiler-style CSVs (Image + compartments) to ``out_dir``.

    Args:
        cells: The single-cell truth frame from :func:`simulate_single_cells`.
        augmented_table: The site-level table used to build the Image CSV.
        out_dir: Directory to write the CSVs into (created if needed).

    Returns:
        Mapping of table name (``Image``/``Cells``/``Cytoplasm``/``Nuclei``) to
        the written CSV path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells_per_site = cells.height // augmented_table.height

    written: dict[str, Path] = {}
    image_path = out / "Image.csv"
    _image_table(augmented_table, cells_per_site).write_csv(image_path)
    written["Image"] = image_path

    for compartment in schema.COMPARTMENTS:
        path = out / f"{compartment}.csv"
        _compartment_table(cells, compartment).write_csv(path)
        written[compartment] = path
    return written


def generate_synthetic_dataset(
    augmented_table: pl.DataFrame,
    paths: DataPaths | None = None,
    *,
    cells_per_site: int = DEFAULT_CELLS_PER_SITE,
    seed: int = DEFAULT_SEED,
    signal: SignalConfig | None = None,
) -> tuple[pl.DataFrame, Path]:
    """Simulate single cells and write CellProfiler CSVs under ``interim``.

    Args:
        augmented_table: Site-level table with patient metadata attached.
        paths: Data locations. Defaults to :func:`get_data_paths`.
        cells_per_site: Cells generated per site.
        seed: RNG seed.
        signal: Signal strengths / feature-correlation knobs.

    Returns:
        The ``(truth_cells_frame, csv_directory)`` pair.
    """
    paths = paths or get_data_paths()
    cells = simulate_single_cells(
        augmented_table, cells_per_site=cells_per_site, seed=seed, signal=signal
    )
    csv_dir = paths.interim / "cellprofiler"
    write_cellprofiler_csvs(cells, augmented_table, csv_dir)
    return cells, csv_dir
