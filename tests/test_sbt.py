
import numpy as np
from numpy.polynomial.polynomial import polyval
import pytest
from scipy.special import gammaln
from scipy.special import spherical_jn

from ffbt.sbt import *


def test_continuousSphericalBesselTransform_bettergrid():
    """Test convergence for increasingly precise grid"""
    rtol = 1e-2
    NREPEAT = 10
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


def test_continuousSphericalBesselTransform_analytic1():
    """Test continuous SBT against analytic result"""
    rtol = 2e-2
    NREPEAT = 2
    for i in range(NREPEAT):
        Nr = np.random.randint(50000, 100000)
        r_grid = np.logspace(-10, 3, Nr)
        k = np.linspace(1e-1, 1, 3)
        s = np.random.uniform(-3, -2, 1)
        f_r = r_grid**s
        ell = np.random.randint(0, 3)
        flk = continuousSphericalBesselTransform(f_r, r_grid, k, ell)
        x1 = (3 + ell + s) / 2.
        x2 = (ell - s) / 2.
        flk_truth = 2.**(s+1.5) / k**(s+3) * np.exp(gammaln(x1) - gammaln(x2))
        print(s, ell, flk/flk_truth)
        np.testing.assert_allclose(flk, flk_truth, rtol=rtol)


def test_rootsSphericalBesselFunctions():
    atol = 1e-12
    N = 100
    L = 100
    roots = rootsSphericalBesselFunctions(L, N)
    ev1 = np.zeros((N, ))
    for ell in range(L):
        ev2 = spherical_jn(ell, roots[ell, :])
        np.testing.assert_allclose(ev1, ev2, atol=atol)


def test_discreteSphericalBesselTransform_loop():
    """Test continuous SBT against analytic result"""
    rtol = 2e-2
    NREPEAT = 2
    for i in range(NREPEAT):
        R = np.random.uniform(1e0, 1e3, 1)
        ell = np.random.randint(0, 10)
        Nr = np.random.randint(5000, 10000)
        N = np.random.randint(5, 100)
        r_grid = np.logspace(-4, np.log10(R), Nr)
        f_l_kln_t = np.random.uniform(0, 1, N)
        kln_nodes, cln_norms = constructGridAndNorms(R, N, ell)
        f_r = inverseDiscreteSphericalBesselTransform(
            f_l_kln_t, kln_nodes, cln_norms, r_grid, ell)
        f_l_kln = forwardDiscreteSphericalBesselTransform(
            f_r, r_grid, ell, N, kln_nodes=kln_nodes, cln_norms=cln_norms)
        print(ell, R, f_l_kln/f_l_kln_t)
        np.testing.assert_allclose(f_l_kln_t, f_l_kln, rtol=rtol)
