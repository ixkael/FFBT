
import numpy as np
from numpy.polynomial.polynomial import polyval
import pytest
from scipy.special import gammaln

from ffbt.sbt import continuousSphericalBesselTransform

rtol = 1e-2
NREPEAT = 10


def test_continuousSphericalBesselTransform_bettergrid():
    """Test convergence for increasingly precise grid"""
    for i in range(NREPEAT):
        P = np.random.randint(10, 20)
        coefs = np.random.uniform(0, 1, P)

        def f(r):
            return polyval(r, coefs)
        Nk = np.random.randint(10, 20)
        Nr = np.random.randint(1000, 2000)
        ell = np.random.randint(10, 20)
        k_nodes = np.random.uniform(0, 1, Nk)
        r_grid1 = np.linspace(0, 1, Nr)
        r_grid2 = np.linspace(0, 1, Nr*10)
        f_r1 = f(r_grid1)
        f_r2 = f(r_grid2)
        flk1 = continuousSphericalBesselTransform(f_r1, r_grid1, k_nodes, ell)
        flk2 = continuousSphericalBesselTransform(f_r2, r_grid2, k_nodes, ell)

        np.testing.assert_allclose(flk1, flk2, rtol=rtol)


def test_continuousSphericalBesselTransform_analytic():
    """Test continuous SBT against analytic result"""
    rtol = 2e-2
    NREPEAT = 5
    for i in range(NREPEAT):
        Nr = np.random.randint(50000, 100000)
        r_grid = np.logspace(-10, 3, Nr)
        k = np.linspace(1e-1, 1, 3)
        s = np.random.uniform(-3, -2, 1)
        f_r = r_grid**s
        ell = np.random.randint(0, 3)
        f0k = continuousSphericalBesselTransform(f_r, r_grid, k, ell)
        x1 = (3 + ell + s) / 2.
        x2 = (ell - s) / 2.
        f0k_truth = 2.**(s+1.5) / k**(s+3) * np.exp(gammaln(x1) - gammaln(x2))
        print(s, ell, f0k/f0k_truth)
        np.testing.assert_allclose(f0k, f0k_truth, rtol=rtol)
