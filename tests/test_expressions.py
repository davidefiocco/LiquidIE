"""Tests for the SymPy-based expression parsing (closures and potentials)."""

import numpy as np
import pytest

from liquidie.expressions import (
    CLOSURE_SYMBOLS,
    HARD_CORE_HEIGHT,
    KNOWN_CLOSURES,
    apply_closure_vec,
    build_closure,
    build_expression,
    build_potential,
    generate_potential_grid,
)


class TestBuildExpression:
    def test_known_name_expanded(self):
        fn = build_expression("PY", KNOWN_CLOSURES, CLOSURE_SYMBOLS)
        assert callable(fn)

    def test_raw_expression(self):
        fn = build_expression("r + phi", KNOWN_CLOSURES, CLOSURE_SYMBOLS)
        assert callable(fn)

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="unknown symbol"):
            build_expression("r + unknown_var", KNOWN_CLOSURES, CLOSURE_SYMBOLS)

    def test_malformed_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            build_expression("r @@@ phi", KNOWN_CLOSURES, CLOSURE_SYMBOLS)


class TestClosures:
    def test_py_closure_values(self):
        fn = build_closure("PY")
        r = np.array([1.0, 2.0])
        gamma_r = np.array([0.1, 0.2])
        phi = np.array([0.5, 0.3])
        inv_t = 1.0
        # (r + gamma_r) * (exp(-inv_t * phi) - 1)
        expected = (r + gamma_r) * (np.exp(-inv_t * phi) - 1)
        # fn args: (gamma_r, inv_t, phi, r) — sorted alphabetical
        result = fn(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result, expected)

    def test_hnc_closure_values(self):
        fn = build_closure("HNC")
        r = np.array([1.0, 2.0])
        gamma_r = np.array([0.1, 0.2])
        phi = np.array([0.5, 0.3])
        inv_t = 1.0
        expected = r * np.exp(-inv_t * phi + gamma_r / r) - gamma_r - r
        result = fn(gamma_r, inv_t, phi, r)
        np.testing.assert_allclose(result, expected)

    def test_custom_closure(self):
        fn = build_closure("r * phi")
        r = np.array([2.0, 3.0])
        result = fn(0.0, 0.0, np.array([4.0, 5.0]), r)
        np.testing.assert_allclose(result, r * np.array([4.0, 5.0]))

    def test_apply_closure_vec(self):
        fn = build_closure("PY")
        r = np.array([0.0, 1.0, 2.0])
        gamma_r = np.zeros((3, 1, 1))
        phi = np.ones((3, 1, 1)) * 0.5
        cr = apply_closure_vec(fn, r, gamma_r, phi, 1.0)
        assert cr.shape == (3, 1, 1)


class TestPotentials:
    def test_hard_sphere(self):
        fn = build_potential("hard_sphere")
        r = np.array([0.3, 0.5, 0.8, 1.0, 1.5])
        # epsilon=1, r=..., sigma=1.0
        result = fn(1.0, r, 1.0)
        assert result[0] == HARD_CORE_HEIGHT  # r < sigma
        assert result[-1] == 0.0  # r >= sigma

    def test_lennard_jones(self):
        fn = build_potential("lennard_jones")
        r = np.array([1.0, 2.0])
        sigma = 1.0
        epsilon = 1.0
        expected = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        result = fn(epsilon, r, sigma)
        np.testing.assert_allclose(result, expected)

    def test_custom_potential(self):
        fn = build_potential("epsilon * (sigma / r)**12")
        r = np.array([1.0, 2.0])
        result = fn(1.0, r, 1.0)
        np.testing.assert_allclose(result, 1.0 / r**12)

    def test_generate_potential_grid(self):
        fn = build_potential("hard_sphere")
        r = np.linspace(0.1, 3.0, 50)
        phi = generate_potential_grid(fn, r, [1.0], [1.0], n_species=1)
        assert phi.shape == (50, 1, 1)
        assert phi[0, 0, 0] == HARD_CORE_HEIGHT  # r=0.1 < sigma=1.0
        assert phi[-1, 0, 0] == 0.0  # r=3.0 >= sigma=1.0
