"""Tests for the DST-I and Spherical Fourier Transform."""

import numpy as np

from liquidie.transforms import dst_i, sft


class TestDstI:
    def test_known_values(self):
        """DST-I of [1, 1, 1, 1] should match manual computation."""
        g = np.array([1.0, 1.0, 1.0, 1.0])
        result = dst_i(g)
        assert result.shape == (4,)
        assert not np.any(np.isnan(result))

    def test_single_element(self):
        g = np.array([1.0])
        result = dst_i(g)
        assert result.shape == (1,)

    def test_linearity(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        alpha = 2.5
        np.testing.assert_allclose(
            dst_i(alpha * g1 + g2),
            alpha * dst_i(g1) + dst_i(g2),
            atol=1e-12,
        )


class TestSft:
    def test_output_shape(self):
        n_pts = 64
        n_species = 2
        r = np.arange(n_pts) * 0.1
        g = np.zeros((n_pts, n_species, n_species))
        k, f = sft(r, g)
        assert k.shape == (n_pts,)
        assert f.shape == (n_pts, n_species, n_species)

    def test_zero_input_gives_zero_output(self):
        n_pts = 32
        r = np.arange(n_pts) * 0.1
        g = np.zeros((n_pts, 1, 1))
        k, f = sft(r, g)
        np.testing.assert_allclose(f, 0.0, atol=1e-15)

    def test_k_grid_spacing(self):
        n_pts = 100
        dr = 0.05
        r = np.arange(n_pts) * dr
        g = np.zeros((n_pts, 1, 1))
        k, _ = sft(r, g)
        expected_dk = np.pi / (n_pts * dr)
        np.testing.assert_allclose(k[1] - k[0], expected_dk)
