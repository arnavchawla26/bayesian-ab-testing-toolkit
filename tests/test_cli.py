import subprocess
import sys


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "bayesab.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_analyze_runs_and_reports_decision():
    result = _run(
        [
            "analyze",
            "--control",
            "120",
            "2000",
            "--treatment",
            "variant_b",
            "220",
            "2000",
            "--seed",
            "1",
            "--mc-samples",
            "20000",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "leader:" in result.stdout
    assert "decision:" in result.stdout
    assert "variant_b" in result.stdout


def test_analyze_supports_multiple_treatments():
    result = _run(
        [
            "analyze",
            "--control",
            "100",
            "2000",
            "--treatment",
            "b",
            "130",
            "2000",
            "--treatment",
            "c",
            "90",
            "2000",
            "--seed",
            "2",
            "--mc-samples",
            "20000",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "b" in result.stdout
    assert "c" in result.stdout


def test_analyze_hdi_flag_runs():
    result = _run(
        [
            "analyze",
            "--control",
            "30",
            "1000",
            "--treatment",
            "b",
            "45",
            "1000",
            "--hdi",
            "--seed",
            "3",
            "--mc-samples",
            "20000",
        ]
    )
    assert result.returncode == 0, result.stderr


def test_analyze_rejects_invalid_counts():
    result = _run(
        [
            "analyze",
            "--control",
            "3000",
            "2000",  # successes > trials
            "--treatment",
            "b",
            "220",
            "2000",
        ]
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""


def test_simulate_runs_and_reports_decision():
    result = _run(
        [
            "simulate",
            "--true-control-rate",
            "0.05",
            "--true-treatment-rate",
            "0.15",
            "--batch-size",
            "300",
            "--max-trials",
            "50000",
            "--seed",
            "1",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "decision:" in result.stdout
    assert "chosen variant:" in result.stdout


def test_no_command_prints_usage_and_fails():
    result = _run([])
    assert result.returncode != 0


def test_help_flag_exits_cleanly():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "analyze" in result.stdout
    assert "simulate" in result.stdout
