"""
Command-line interface for the mock patient-profile workflow.

Exposes the end-to-end pipeline as a single command so a new contributor can,
after ``uv sync``, reproduce the whole workflow with::

    uv run mock-patient-profile run

which downloads the small BBBC021 metadata and writes single cells, QC reports,
per-patient morphology profiles, and the integrated multi-omic table under the
data directory.
"""

from __future__ import annotations

from importlib.metadata import version

import fire

from .paths import get_data_paths
from .pipeline import PipelineConfig, run_pipeline
from .synthetic import SignalConfig


def _print_summary(summary: dict[str, object]) -> None:
    """Print a human-readable run summary."""
    print("\nmock-patient-profile pipeline complete")
    print(
        f"  plates={summary['n_plates']} wells={summary['n_wells']} "
        f"cells={summary['n_cells']} patients={summary['n_patients']}"
    )
    print(
        f"  selected features={summary['n_features_selected']} "
        f"outlier flags={summary['n_outlier_flags']} "
        f"integrated rows={summary['n_integrated_rows']}"
    )
    print("  outputs:")
    for name, path in summary["outputs"].items():  # type: ignore[union-attr]
        print(f"    - {name}: {path}")


class Cli:
    """mock-patient-profile commands."""

    def run(  # noqa: PLR0913
        self,
        data_dir: str | None = None,
        n_plates: int = 2,
        cells_per_site: int = 15,
        n_patients: int = 8,
        seed: int = 0,
        normalize_method: str = "standardize",
        disease_plate_confounding: float = 0.0,
        feature_correlation: float = 0.0,
        batch_weight: float = 0.4,
        force_download: bool = False,
    ) -> None:
        """Run the full BBBC021 -> patient-profile pipeline.

        Args:
            data_dir: Data root (defaults to ``$MOCK_PATIENT_PROFILE_DATA`` or
                ``./data``).
            n_plates: Number of BBBC021 plates to include.
            cells_per_site: Synthetic cells generated per imaging site.
            n_patients: Size of the synthetic patient cohort.
            seed: Random seed for reproducibility.
            normalize_method: pycytominer normalization method.
            disease_plate_confounding: Disease<->plate confounding in [0, 1]
                (0 = balanced/easy; 1 = disease confounded with batch).
            feature_correlation: Within-compartment feature correlation in
                [0, 1) for realism (0 = independent features).
            batch_weight: Strength of the per-plate batch signal (raise above the
                disease weight, ~0.9, to make batch overwhelm biology).
            force_download: Force re-download of BBBC021 metadata.
        """
        paths = get_data_paths(data_dir)
        config = PipelineConfig(
            n_plates=n_plates,
            cells_per_site=cells_per_site,
            n_patients=n_patients,
            seed=seed,
            normalize_method=normalize_method,
            disease_plate_confounding=disease_plate_confounding,
            signal=SignalConfig(
                weight_plate=batch_weight,
                feature_correlation=feature_correlation,
            ),
        )
        summary = run_pipeline(paths, config=config, force_download=force_download)
        _print_summary(summary)

    def download(self, data_dir: str | None = None, force: bool = False) -> None:
        """Download only the BBBC021 metadata CSVs into the data directory.

        Args:
            data_dir: Data root.
            force: Re-download even if cached.
        """
        from .bbbc021 import download_bbbc021_metadata

        paths = get_data_paths(data_dir, ensure=True)
        downloaded = download_bbbc021_metadata(paths, force=force)
        for name, path in downloaded.items():
            print(f"{name}: {path}")

    def info(self, data_dir: str | None = None) -> None:
        """Print package version and resolved data directory.

        Args:
            data_dir: Data root.
        """
        paths = get_data_paths(data_dir)
        print(f"mock-patient-profile {version('mock-patient-profile')}")
        print(f"data root: {paths.root}")


def trigger() -> None:
    """Entry point for the ``mock-patient-profile`` console script."""
    fire.Fire(Cli)
