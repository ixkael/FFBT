
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
            f_r, r_grid, ell, kln_nodes, cln_norms)
        print(ell, R, f_l_kln/f_l_kln_t)
        np.testing.assert_allclose(f_l_kln_t, f_l_kln, rtol=rtol)


def test_discreteSphericalBesselTransform_doublediscrete():
    """Test continuous SBT against analytic result"""
    rtol = 2e-2
    NREPEAT = 5
    for i in range(NREPEAT):
        R = np.random.uniform(1e0, 1e3, 1)
        ell = np.random.randint(0, 10)
        N = np.random.randint(50, 200)
        kln_nodes, cln_norms = constructGridAndNorms(R, N, ell)
        K = kln_nodes[-1]
        rln_nodes, dln_norms = constructGridAndNorms(K, N, ell)
        f_l_kln_t = np.random.uniform(0, 1, N)
        f_l_kln_t[-1] = 0
        f_r = inverseDiscreteSphericalBesselTransform(
            f_l_kln_t, kln_nodes, cln_norms, rln_nodes, ell)
        f_l_kln = inverseDiscreteSphericalBesselTransform(
            f_r, rln_nodes, dln_norms, kln_nodes, ell)
        print(ell, R, f_l_kln/f_l_kln_t)
        np.testing.assert_allclose(f_l_kln_t[:-1], f_l_kln[:-1], rtol=rtol)


def test_discreteSphericalBesselTransform_fast():
    """Test continuous SBT against analytic result"""
    rtol = 2e-2
    NREPEAT = 2
    for i in range(NREPEAT):
        R = np.random.uniform(1e0, 1e3, 1)
        ell = np.random.randint(0, 5)
        N = np.random.randint(100, 500)
        kln_nodes, cln_norms = constructGridAndNorms(R, N, ell)
        K = kln_nodes[-1]
        f_l_kln_t = np.random.uniform(0, 1, N)
        f_l_kln_t[-1] = 0
        mat = fastTransformMatrix(ell, ell, N)
        f_r = np.dot(mat, f_l_kln_t) / R**3
        f_l_kln = np.dot(mat, f_r) / K**3
        print(ell, K, f_l_kln/f_l_kln_t)
        np.testing.assert_allclose(f_l_kln[:-1], f_l_kln_t[:-1], rtol=rtol)


def test_couplingMatrix():
    """Test coupling matrix"""
    rtol = 1e-0
    NREPEAT = 2
    for i in range(NREPEAT):
        R = np.random.uniform(1e0, 1e1, 1)
        Nr = np.random.randint(900, 1000)
        N = np.random.randint(50, 100)

        r_grid = np.logspace(-8, np.log10(R), Nr)
        f_l1_k_t = np.random.uniform(0, 1, N)
        f_l1_k_t[-1] = 0

        ell1 = np.random.randint(0, 3)
        ell2 = np.random.randint(0, 3)
        kln1, cln1 = constructGridAndNorms(R, N, ell1)
        K1 = kln1[-1]
        kln2, cln2 = constructGridAndNorms(R, N, ell2)
        K2 = kln2[-1]
        rl1, dl1 = constructGridAndNorms(K1, N, ell1)
        rl2, dl2 = constructGridAndNorms(K2, N, ell2)
        f_l1_r = rl1*1
        f_l2_r = rl2*1

        mat11 = fastTransformMatrix(ell1, ell1, N)
        f_l1_k = np.dot(mat11, f_l1_r) / K1**3
        mat22 = fastTransformMatrix(ell2, ell2, N)
        f_l2_k = np.dot(mat22, f_l2_r) / K2**3

        cou12 = orderCouplingMatrix(ell1, ell2, N, Nr)
        f_l1_k2 = np.dot(cou12, f_l2_k)
        print(R, ell1, ell2, f_l1_k2[:-1]/f_l1_k[:-1])
        np.testing.assert_allclose(f_l1_k2[:-1], f_l1_k[:-1], rtol=rtol)
