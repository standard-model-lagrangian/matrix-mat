"""CLI fixes (0.2.2): config resolution, flag validation, aliases, exit codes."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import matrix_mat
from matrix_mat import cli
from matrix_mat.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert matrix_mat.__version__ in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["bf", "run"], ["bf", "preflight"], ["bf", "plots"], ["if", "preflight"], ["if", "sweep"],
])
def test_explicit_missing_config_exits_2(argv, tmp_path, capsys):
    code = main(argv + ["-c", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "nope.yaml" in capsys.readouterr().err


def test_default_config_resolves_against_repo_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = cli.resolve_config_path(None, "spheroid_brightfield.yaml")
    assert p == REPO_ROOT / "configs" / "spheroid_brightfield.yaml"


def test_absent_default_config_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_DIR", tmp_path / "none")
    with caplog.at_level(logging.WARNING, logger="matrix_mat.cli"):
        assert cli.resolve_config_path(None, "spheroid_brightfield.yaml") is None
    assert "defaults" in caplog.text.lower()


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "abc"])
def test_pixel_size_must_be_positive(bad):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bf", "run", "--pixel-size", bad])


def test_device_choices():
    p = build_parser()
    for d in ("auto", "cpu", "cuda", "mps", "cuda:1"):
        assert p.parse_args(["bf", "run", "--device", d]).device == d
    with pytest.raises(SystemExit):
        p.parse_args(["bf", "run", "--device", "tpu"])


def test_no_abbreviations():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bf", "run", "--over"])


@pytest.mark.parametrize("argv", [
    ["bf", "preflight"], ["bf", "run"], ["bf", "plots"], ["if", "preflight"], ["if", "sweep"],
    ["if", "compare"], ["if", "compare-prepost"], ["if", "compare-biological"],
])
def test_output_dir_flags_everywhere(argv):
    p = build_parser()
    a = p.parse_args(argv + ["-o", "X"])
    b = p.parse_args(argv + ["--output-dir", "X"])
    dest = "runs_dir" if argv[1] == "sweep" else "output_dir"
    assert getattr(a, dest) == getattr(b, dest) == "X"


def test_verbosity_after_subcommand():
    a = build_parser().parse_args(["bf", "run", "-v"])
    assert a.verbose is True
    b = build_parser().parse_args(["-q", "if", "compare"])
    assert b.quiet is True
    c = build_parser().parse_args(["if", "compare"])
    assert c.verbose is False and c.quiet is False


def test_bf_preflight_has_input_dir():
    a = build_parser().parse_args(["bf", "preflight", "--input-dir", "D"])
    assert a.input_dir == "D"


def test_input_dir_resolution_generic(tmp_path):
    (tmp_path / "Spheroid Day0 999999").mkdir()
    (tmp_path / "Spheroid D7 999999").mkdir()
    t0, t7 = cli.resolve_input_subdirs(tmp_path)
    assert t0.name == "Spheroid Day0 999999" and t7.name == "Spheroid D7 999999"
    (tmp_path / "Day0").mkdir()
    t0, _ = cli.resolve_input_subdirs(tmp_path)
    assert t0.name == "Day0"  # exact names win over pattern matches
    t0, t7 = cli.resolve_input_subdirs(tmp_path, t0_subdir="Spheroid Day0 999999", t7_subdir="Spheroid D7 999999")
    assert t0.name == "Spheroid Day0 999999"


def test_input_dir_resolution_ambiguous(tmp_path):
    (tmp_path / "A Day0 1").mkdir()
    (tmp_path / "B Day0 2").mkdir()
    (tmp_path / "Day7").mkdir()
    with pytest.raises(ValueError, match="ambiguous"):
        cli.resolve_input_subdirs(tmp_path)


def test_full_dataset_overrides_yaml_sample(monkeypatch, tmp_path):
    import matrix_mat.brightfield.run_pipeline as rp

    seen = {}

    def fake(config=None, sample_n=None):
        seen["config_sample"] = config.sample_n
        seen["sample_n"] = sample_n
        return {"status": "SUCCESS"}

    monkeypatch.setattr(rp, "run_spheroid_pipeline", fake)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("sample_n: 4\n")
    assert main(["bf", "run", "-c", str(cfg), "--full-dataset", "-o", str(tmp_path / "o")]) == 0
    assert seen == {"config_sample": None, "sample_n": None}


def test_pixel_size_override_reaches_config(monkeypatch, tmp_path):
    import matrix_mat.brightfield.run_pipeline as rp

    seen = {}
    monkeypatch.setattr(rp, "run_spheroid_pipeline",
                        lambda config=None, sample_n=None: seen.setdefault("c", config) and {"status": "SUCCESS"})
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")
    main(["bf", "run", "-c", str(cfg), "--pixel-size", "0.5", "--device", "cpu", "-o", str(tmp_path / "o")])
    assert seen["c"].cli_pixel_size_um == 0.5
    assert seen["c"].segmentation.device == "cpu"


def test_bf_plots_fails_when_figures_fail(monkeypatch, tmp_path, capsys):
    import matrix_mat.reporting.presentation_plots as pp

    (tmp_path / "objects.csv").write_text("image_id\nx\n")
    (tmp_path / "pairs.csv").write_text("pair_key\nx\n")
    import matrix_mat.reporting.visualize as vz

    monkeypatch.setattr(vz, "generate_all_figures", lambda **kw: None)

    def boom(**kw):
        raise RuntimeError("figure exploded")

    monkeypatch.setattr(pp, "generate_all_presentation_figures", boom)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")
    code = main(["bf", "plots", "-c", str(cfg), "-o", str(tmp_path)])
    assert code == 1
    assert "figure exploded" in capsys.readouterr().err


def test_bf_plots_uses_config_output_dir(monkeypatch, tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    out = tmp_path / "from_yaml"
    out.mkdir()
    cfg.write_text(f"output_dir: '{out}'\n")
    code = main(["bf", "plots", "-c", str(cfg)])
    assert code == 1
    assert str(out / "objects.csv") in capsys.readouterr().err


@pytest.mark.parametrize("sub", ["compare", "compare-prepost"])
def test_if_compare_missing_runs_dir_is_clean_error(sub, tmp_path, capsys):
    code = main(["if", sub, "--runs", str(tmp_path / "missing")])
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "missing" in err


def test_compare_biological_missing_run_dir_is_clean_error(tmp_path, capsys):
    code = main(["if", "compare-biological", "--run-dir", str(tmp_path / "missing")])
    assert code == 1


def test_compare_prepost_biological_multiple_matches_errors(monkeypatch, tmp_path, capsys):
    import matrix_mat.if_density.comparator_prepost as cp

    runs = tmp_path / "runs"
    (runs / "104_PP11_full").mkdir(parents=True)
    (runs / "105_PP11_subset").mkdir(parents=True)
    monkeypatch.setattr(cp, "run_comparator_prepost", lambda **kw: runs)
    assert main(["if", "compare-prepost", "--runs", str(runs), "--biological"]) == 1
    assert "105_PP11_subset" in capsys.readouterr().err


def test_compare_biological_run_pattern(monkeypatch, tmp_path):
    import matrix_mat.if_density.comparator_biological as cb

    runs = tmp_path / "runs"
    (runs / "200_custom").mkdir(parents=True)
    seen = {}
    monkeypatch.setattr(cb, "generate_all_if_biological_figures",
                        lambda run_dir, output_dir=None: seen.setdefault("r", run_dir))
    assert main(["if", "compare-biological", "--runs", str(runs), "--run-pattern", "*custom*"]) == 0
    assert Path(seen["r"]).name == "200_custom"


def test_sample_help_describes_domains():
    p = build_parser()
    sub = {a.dest: a for a in p._subparsers._group_actions}
    bf = sub["domain"].choices["bf"]._subparsers._group_actions[0].choices
    iff = sub["domain"].choices["if"]._subparsers._group_actions[0].choices
    bf_help = next(a.help for a in bf["run"]._actions if a.dest == "sample")
    if_help = next(a.help for a in iff["sweep"]._actions if a.dest == "sample")
    assert "pair" in bf_help.lower()
    assert "field" in if_help.lower()


def test_if_sweep_dry_run_passthrough(monkeypatch):
    import matrix_mat.if_density.sweep as sweep_mod

    seen = {}
    monkeypatch.setattr(sweep_mod, "run_sweep", lambda **kw: seen.update(kw) or {"status": "SUCCESS"})
    assert main(["if", "sweep", "-c", str(REPO_ROOT / "configs" / "sweep.yaml"), "--dry-run"]) == 0
    assert seen["dry_run"] is True


def test_bf_run_dry_run_lists_without_running(monkeypatch, tmp_path, capsys):
    import matrix_mat.brightfield.run_pipeline as rp

    monkeypatch.setattr(rp, "run_spheroid_pipeline", lambda **kw: pytest.fail("must not run"))
    code = main(["bf", "run", "-c", str(REPO_ROOT / "configs" / "spheroid_brightfield.yaml"),
                 "--t0-dir", str(REPO_ROOT / "tests/fixtures/synthetic_dataset/Day0"),
                 "--t7-dir", str(REPO_ROOT / "tests/fixtures/synthetic_dataset/Day7"),
                 "-o", str(tmp_path / "o"), "--dry-run"])
    assert code == 0
    assert "Dry run:" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_if_sweep_strict_pairing_passthrough(monkeypatch):
    import matrix_mat.if_density.sweep as sweep_mod

    seen = {}
    monkeypatch.setattr(sweep_mod, "run_sweep", lambda **kw: seen.update(kw) or {"status": "SUCCESS"})
    assert main(["if", "sweep", "-c", str(REPO_ROOT / "configs" / "sweep.yaml"), "--strict-pairing"]) == 0
    assert seen["strict_pairing"] is True and seen["dry_run"] is False


def test_if_compare_run_names_and_scope(monkeypatch, tmp_path):
    import matrix_mat.if_density.comparator as comp
    import matrix_mat.if_density.comparator_prepost as cp

    seen = {}
    monkeypatch.setattr(comp, "generate_cross_run_comparison", lambda **kw: seen.setdefault("c", kw))
    monkeypatch.setattr(cp, "run_comparator_prepost", lambda **kw: seen.setdefault("p", kw))
    assert main(["if", "compare", "--runs", str(tmp_path), "--run-names", "a", "b", "--scope", "subset"]) == 0
    assert seen["c"]["run_names"] == ["a", "b"] and seen["c"]["scope"] == "subset"
    assert main(["if", "compare-prepost", "--runs", str(tmp_path), "--scope", "subset"]) == 0
    assert seen["p"]["scope"] == "subset"
