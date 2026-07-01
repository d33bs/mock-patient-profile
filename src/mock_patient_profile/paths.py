"""
Data directory layout for the mock patient-profile workflow.

The workflow reads and writes everything under a single data root with a
conventional ``raw`` / ``interim`` / ``processed`` / ``reports`` split. The
root is resolved (in priority order) from an explicit argument, the
``MOCK_PATIENT_PROFILE_DATA`` environment variable, or a ``data/`` directory in
the current working directory. The whole tree is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Environment variable used to override the data root.
DATA_ENV_VAR = "MOCK_PATIENT_PROFILE_DATA"

#: Default data root, relative to the current working directory.
DEFAULT_DATA_DIRNAME = "data"


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations for each stage of the workflow.

    Attributes:
        root: The data root directory.
    """

    root: Path

    @property
    def raw(self) -> Path:
        """Downloaded, read-only source data (e.g. BBBC021 metadata CSVs)."""
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        """Intermediate artifacts (e.g. synthetic CellProfiler outputs)."""
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        """Canonical Parquet outputs (single cells, profiles, mock tables)."""
        return self.root / "processed"

    @property
    def reports(self) -> Path:
        """Human-facing outputs (QC reports, summaries)."""
        return self.root / "reports"

    def ensure(self) -> DataPaths:
        """Create all stage directories if they do not exist.

        Returns:
            ``self``, to allow fluent use.
        """
        for directory in (self.raw, self.interim, self.processed, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def get_data_paths(
    root: str | os.PathLike[str] | None = None,
    *,
    ensure: bool = False,
) -> DataPaths:
    """Resolve the :class:`DataPaths` for the workflow.

    Args:
        root: Explicit data root. When ``None``, falls back to the
            ``MOCK_PATIENT_PROFILE_DATA`` environment variable and then to
            ``./data``.
        ensure: When ``True``, create the directory tree before returning.

    Returns:
        The resolved :class:`DataPaths`.
    """
    if root is None:
        root = os.environ.get(DATA_ENV_VAR) or Path.cwd() / DEFAULT_DATA_DIRNAME
    paths = DataPaths(Path(root).expanduser().resolve())
    if ensure:
        paths.ensure()
    return paths
