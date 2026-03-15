"""Discrete Sine Transform (type I) and Spherical Fourier Transform.

The spherical Fourier transform converts radial functions f(r) to their
reciprocal-space counterparts f(k) for isotropic 3-D systems, using the
relation  f(k) = 4*pi * integral[ r * f(r) * sin(kr) / k , r=0..inf ].
In practice this is computed via a DST-I on a uniform grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft


def dst_i(g: NDArray[np.floating]) -> NDArray[np.floating]:
    """Type-I Discrete Sine Transform via FFT.

    Parameters
    ----------
    g : 1-D array
        Input signal (excluding the zero-padded boundaries).

    Returns
    -------
    1-D array of same length as *g*.
    """
    worm = np.concatenate(([0.0], g, [0.0], -g[::-1]))
    return -np.imag(fft(worm)[1 : len(g) + 1]) / 2.0


def sft(
    r: NDArray[np.floating],
    g: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Spherical Fourier Transform on a uniform radial grid.

    Parameters
    ----------
    r : 1-D array, shape (N,)
        Radial grid points (uniform spacing assumed).
    g : 3-D array, shape (N, n_species, n_species)
        Function values on the radial grid for each species pair.

    Returns
    -------
    k : 1-D array, shape (N,)
        Reciprocal-space grid.
    f : 3-D array, shape (N, n_species, n_species)
        Transformed function values.
    """
    dr = r[1] - r[0]
    n_pts = len(r)
    dk = np.pi / (n_pts * dr)
    k = np.arange(n_pts) * dk

    n_species = g.shape[1]
    f = np.zeros_like(g)
    for i in range(n_species):
        for j in range(n_species):
            f[:, i, j] = np.concatenate(([0.0], dst_i(g[1:, i, j]) * dr))

    return k, f
