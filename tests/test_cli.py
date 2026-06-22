"""
Tests for the command-line interface.
"""

import subprocess

from mock_patient_profile.cli import Cli


def test_info_prints_version_and_data_root(capsys) -> None:
    Cli().info()
    out = capsys.readouterr().out
    assert "mock-patient-profile" in out
    assert "data root" in out


def test_console_script_info_entry_point() -> None:
    result = subprocess.run(
        ["uv", "run", "mock-patient-profile", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mock-patient-profile" in result.stdout
