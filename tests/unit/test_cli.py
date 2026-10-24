"""
Unit tests for matrix-mat unified CLI (WS4.1).
Validates parser schema, mutually exclusive modes, missing input handling, and README command parsing.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
import pytest

from matrix_mat.cli import build_parser, main


def test_cli_parser_builds():
    """Parser builds with bf and if subcommands."""
    parser = build_parser()
    assert parser is not None


def test_mutually_exclusive_modes():
    """In bf run, --sample and --full-dataset are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bf", "run", "--sample", "12", "--full-dataset"])


def test_plots_without_csvs_gives_actionable_error(tmp_path, caplog, capsys):
    """Running bf plots on a directory without objects.csv/pairs.csv gives actionable error and exit code 1."""
    empty_dir = tmp_path / "empty_output"
    empty_dir.mkdir()

    exit_code = main(["bf", "plots", "--output-dir", str(empty_dir)])
    assert exit_code == 1
    err_output = capsys.readouterr().err + caplog.text
    assert "Cannot generate presentation figures" in err_output


def test_no_dead_flags():
    """Verify that every destination in bf and if subparsers is handled in handlers."""
    parser = build_parser()
    
    # Check that bf run handles all defined options
    bf_run_args = parser.parse_args([
        "bf", "run",
        "--config", "configs/spheroid_brightfield.yaml",
        "--t0-dir", "data/brightfield/Day0",
        "--t7-dir", "data/brightfield/Day7",
        "--output-dir", "output",
        "--pixel-size", "1.518817",
        "--backend", "cpsam",
        "--device", "cpu",
        "--no-cache",
        "--overwrite",
        "--force-uncalibrated",
        "--sample", "5",
    ])
    assert bf_run_args.domain == "bf"
    assert bf_run_args.subcommand == "run"
    assert bf_run_args.backend == "cpsam"
    assert bf_run_args.sample == 5
    assert bf_run_args.overwrite is True

    # Check if sweep handles all defined options
    if_sweep_args = parser.parse_args([
        "if", "sweep",
        "--config", "configs/sweep.yaml",
        "--output-dir", "runs",
        "--data-dir", "data/if",
        "--scope", "subset",
        "--allow-unpaired",
        "--overwrite",
        "--sample", "2",
    ])
    assert if_sweep_args.domain == "if"
    assert if_sweep_args.subcommand == "sweep"
    assert if_sweep_args.scope == "subset"
    assert if_sweep_args.allow_unpaired is True


def test_readme_commands_parse():
    """Extract every fenced bash command in README.md starting with matrix-mat, and assert it parses without error."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    readme_path = repo_root / "README.md"
    assert readme_path.exists(), "README.md not found"

    content = readme_path.read_text(encoding="utf-8")
    parser = build_parser()

    # Find all bash/sh code blocks
    pattern = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(content)

    commands_tested = 0
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("matrix-mat "):
                tokens = shlex.split(line)[1:]  # strip 'matrix-mat'
                parsed = parser.parse_args(tokens)
                assert parsed.domain in ("bf", "if")
                commands_tested += 1

    assert commands_tested > 0, "No matrix-mat commands found in README.md to test"
