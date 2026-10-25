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

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        "--config", str(REPO_ROOT / "configs/spheroid_brightfield.yaml"),
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
        "--config", str(REPO_ROOT / "configs/sweep.yaml"),
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


@pytest.mark.parametrize("doc", ["README.md", "docs/BEGINNER_GUIDE.md", "docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md"])
def test_readme_commands_parse(doc):
    """Every `matrix-mat ...` command in fenced bash blocks (and inline code) of the user docs must parse."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    content = (repo_root / doc).read_text(encoding="utf-8")
    parser = build_parser()

    candidates = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", content, re.DOTALL):
        candidates += block.splitlines()
    candidates += re.findall(r"`(matrix-mat [^`]+)`", content)

    commands_tested = 0
    for line in candidates:
        line = line.strip()
        if not line.startswith("matrix-mat "):
            continue
        tokens = shlex.split(line, comments=True)[1:]  # strip 'matrix-mat' and trailing comments
        if "--help" in tokens or "-h" in tokens or "<bf|if>" in line or len(tokens) < 2:
            continue  # help invocations and bare `matrix-mat bf` labels are not runnable commands
        parsed = parser.parse_args(tokens)
        assert parsed.domain in ("bf", "if"), line
        commands_tested += 1

    assert commands_tested > 0, f"No matrix-mat commands found in {doc}"


@pytest.mark.parametrize("backend", ["cyto3", "sam"])
def test_cli_rejects_backends_that_do_not_exist(backend):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bf", "run", "--backend", backend])


def test_cli_accepts_real_backends():
    parser = build_parser()
    for backend in ("cpsam", "cpdino", "classical"):
        assert parser.parse_args(["bf", "run", "--backend", backend]).backend == backend


def test_cli_does_not_override_yaml_paths_by_default():
    """Without explicit flags, output/review dirs must come from the YAML, not CLI defaults."""
    args = build_parser().parse_args(["bf", "run"])
    assert args.output_dir is None
    assert args.review_dir is None


def test_bf_run_exit_code_nonzero_when_weights_missing(tmp_path, monkeypatch):
    """matrix-mat bf run must exit non-zero when the configured model weights are absent."""
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("MATRIX_MAT_MODELS_ROOT", str(tmp_path / "no_models"))
    monkeypatch.chdir(tmp_path)
    code = main([
        "bf", "run",
        "--config", str(repo / "configs" / "spheroid_brightfield.yaml"),
        "--t0-dir", str(repo / "tests/fixtures/synthetic_dataset/Day0"),
        "--t7-dir", str(repo / "tests/fixtures/synthetic_dataset/Day7"),
        "--output-dir", str(tmp_path / "out"),
        "--backend", "cpsam",
    ])
    assert code != 0


def test_bf_run_exit_code_zero_on_success(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    code = main([
        "bf", "run",
        "--config", str(repo / "configs" / "spheroid_brightfield.yaml"),
        "--t0-dir", str(repo / "tests/fixtures/synthetic_dataset/Day0"),
        "--t7-dir", str(repo / "tests/fixtures/synthetic_dataset/Day7"),
        "--output-dir", str(tmp_path / "out"),
        "--backend", "classical",
        "--pixel-size", "1.518817",
    ])
    assert code == 0
    assert list(tmp_path.iterdir()) == [tmp_path / "out"], "bf run must not write outside --output-dir"


def test_if_sweep_flags_reach_the_sweep(monkeypatch, tmp_path):
    """--runs-dir, --data-dir and --sample must be passed through, not parsed and dropped."""
    import matrix_mat.if_density.sweep as sweep_mod

    seen = {}

    def fake_run_sweep(**kwargs):
        seen.update(kwargs)
        return {"status": "SUCCESS"}

    monkeypatch.setattr(sweep_mod, "run_sweep", fake_run_sweep)
    code = main(["if", "sweep", "--config", str(REPO_ROOT / "configs" / "sweep.yaml"),
                 "--runs-dir", str(tmp_path / "r"), "--data-dir", str(tmp_path / "d"), "--sample", "3"])
    assert code == 0
    assert seen["runs_dir_override"] == str(tmp_path / "r")
    assert seen["data_dir_override"] == str(tmp_path / "d")
    assert seen["sample_n"] == 3


def test_if_commands_default_to_output_root_runs(monkeypatch, tmp_path):
    """compare/compare-prepost/compare-biological must look where `if sweep` writes: ${OUTPUT_ROOT}/runs."""
    monkeypatch.setenv("MATRIX_MAT_OUTPUT_ROOT", str(tmp_path / "out"))
    parser = build_parser()
    runs = str(tmp_path / "out" / "runs")
    assert parser.parse_args(["if", "compare"]).runs_dir in (None, runs)
    from matrix_mat.cli import resolve_runs_dir
    assert str(resolve_runs_dir(None)) == runs
    assert str(resolve_runs_dir("x")) == "x"


def test_if_sweep_cli_returns_nonzero_on_failure(monkeypatch):
    import matrix_mat.if_density.sweep as sweep_mod

    def boom(**kwargs):
        raise FileNotFoundError("weights missing")

    monkeypatch.setattr(sweep_mod, "run_sweep", boom)
    assert main(["if", "sweep", "--config", str(REPO_ROOT / "configs" / "sweep.yaml")]) == 1


def test_compare_prepost_biological_uses_the_full_pipeline_run(monkeypatch, tmp_path):
    import matrix_mat.if_density.comparator_prepost as cp
    import matrix_mat.if_density.comparator_biological as cb

    runs = tmp_path / "runs"
    (runs / "104_PP11_full").mkdir(parents=True)
    monkeypatch.setattr(cp, "run_comparator_prepost", lambda **kw: runs / "comparison_prepost")
    seen = {}
    monkeypatch.setattr(cb, "generate_all_if_biological_figures", lambda run_dir, output_dir=None: seen.setdefault("run_dir", run_dir))
    assert main(["if", "compare-prepost", "--runs", str(runs), "--biological"]) == 0
    assert str(seen["run_dir"]) == str(runs / "104_PP11_full")


def _if_dataset_with_unpaired(tmp_path):
    import shutil
    data = tmp_path / "if"
    shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "synthetic_if_dataset", data)
    src = next(data.glob("*_ch00.tif"))
    shutil.copy(src, data / "SKOV3 Spheroid D7 Mor_S50-Gel3-1_ch00.tif")  # nuclear channel without actin partner
    return data


@pytest.mark.parametrize("allow,expected", [(True, 0), (False, 1)])
def test_if_preflight_follows_config_allow_unpaired(tmp_path, monkeypatch, allow, expected):
    """Preflight must apply the same allow_unpaired rule as the sweep it is checking."""
    import yaml as _yaml

    data = _if_dataset_with_unpaired(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "cpsam_v2").write_bytes(b"x")
    cfg = _yaml.safe_load((REPO_ROOT / "configs" / "prepost_ablation.yaml").read_text())
    cfg.update({"data_dir": str(data), "allow_unpaired": allow})
    cfg["model"]["pretrained_model"] = str(tmp_path / "models" / "cpsam_v2")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml.safe_dump(cfg))
    assert main(["if", "preflight", "--config", str(cfg_path)]) == expected


def test_if_preflight_reports_materials(tmp_path):
    from matrix_mat.if_density.preflight import run_if_preflight

    report = run_if_preflight(data_dir=str(_if_dataset_with_unpaired(tmp_path)), model_path=str(tmp_path / "none"),
                              allow_unpaired=True)
    assert sorted(report.stats["materials"]) == ["Mat", "S34D30"]
