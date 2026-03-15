"""Tests for the CLI, I/O round-trip, and restart path."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import click

from liquidie.cli import app
from liquidie.config import Config
from liquidie.solver import solve, write_results

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ornstein-Zernike" in click.unstyle(result.stdout)

    def test_solve_help(self):
        result = runner.invoke(app, ["solve", "--help"])
        assert result.exit_code == 0
        assert "--config" in click.unstyle(result.stdout)

    def test_mct_help(self):
        result = runner.invoke(app, ["mct", "--help"])
        assert result.exit_code == 0
        assert "--config" in click.unstyle(result.stdout)

    def test_solve_runs(self, tmp_path):
        config_path = EXAMPLES_DIR / "hard_sphere_1species.toml"
        out = tmp_path / "results"
        result = runner.invoke(
            app, ["solve", "--config", str(config_path), "--output-dir", str(out)]
        )
        assert result.exit_code == 0, result.stdout
        assert (out / "rdf00.dat").exists()
        assert (out / "s00.dat").exists()
        assert (out / "gamma.dat").exists()


# ---------------------------------------------------------------------------
# I/O round-trip test
# ---------------------------------------------------------------------------


class TestIOroundTrip:
    @pytest.fixture(scope="class")
    def solved(self):
        config = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.02, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        return solve(config)

    def test_roundtrip_single_species(self, solved, tmp_path):
        """write_results -> reload files -> arrays match original."""
        write_results(solved, tmp_path)

        rdf_data = np.loadtxt(tmp_path / "rdf00.dat")
        np.testing.assert_allclose(rdf_data[:, 0], solved.r, atol=1e-12)
        np.testing.assert_allclose(rdf_data[:, 1], solved.rdf[:, 0, 0], atol=1e-12)

        s_data = np.loadtxt(tmp_path / "s00.dat")
        np.testing.assert_allclose(s_data[:, 0], solved.k, atol=1e-12)
        np.testing.assert_allclose(s_data[:, 1], solved.s_k[:, 0, 0], atol=1e-12)

        c_data = np.loadtxt(tmp_path / "c00.dat")
        np.testing.assert_allclose(c_data[:, 1], solved.c_k[:, 0, 0], atol=1e-12)

        h_data = np.loadtxt(tmp_path / "h00.dat")
        np.testing.assert_allclose(h_data[:, 1], solved.h_k[:, 0, 0], atol=1e-12)

    def test_gamma_roundtrip(self, solved, tmp_path):
        """gamma.dat can be reloaded and matches the original gamma_r."""
        write_results(solved, tmp_path)
        gamma_data = np.loadtxt(tmp_path / "gamma.dat")
        r_loaded = gamma_data[:, 0]
        gamma_loaded = gamma_data[:, 1:].reshape(len(r_loaded), 1, 1)
        np.testing.assert_allclose(r_loaded, solved.r, atol=1e-12)
        np.testing.assert_allclose(gamma_loaded, solved.gamma_r, atol=1e-12)


# ---------------------------------------------------------------------------
# Restart path test
# ---------------------------------------------------------------------------


class TestRestart:
    def test_restart_uses_warm_start(self, tmp_path):
        """Solver with restart=True loads gamma.dat as initial guess."""
        config_base = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.02, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
        )
        result_cold = solve(config_base)
        write_results(result_cold, tmp_path)

        config_restart = Config(
            system={"temperature": 1.0, "density": [0.3]},
            grid={"dr": 0.02, "r_max": 20.0},
            potential={
                "expression": "hard_sphere",
                "epsilon": [1.0],
                "sigma": [1.0],
            },
            solver={"closure": "PY", "tolerance": 1e-8},
            restart={"enabled": True, "file": str(tmp_path / "gamma.dat")},
        )
        result_warm = solve(config_restart)

        np.testing.assert_allclose(
            result_warm.rdf[:, 0, 0],
            result_cold.rdf[:, 0, 0],
            atol=1e-6,
            err_msg="Warm-start should converge to same solution as cold start",
        )
