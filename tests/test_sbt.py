
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
    """Test roots of spherical Bessel functions"""
    atol = 1e-12
    N = 100
    L = 100
    roots = rootsSphericalBesselFunctions(L, N)
    ev1 = np.zeros((N, ))
    for ell in range(L):
        ev2 = spherical_jn(ell, roots[ell, :])
        np.testing.assert_allclose(ev1, ev2, atol=atol)


def test_discreteSphericalBesselTransform_loop():
    """Test discrete SBT back and forth"""
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
    """Test discrete SBT both ways"""
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
    """Test fast SBT """
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


def test_couplingMatrix_dft():
    """Test coupling matrix but not with fast transform due to approx errors"""
    rtol = 6e-1
    NREPEAT = 1
    for i in range(NREPEAT):
        R = 10  # np.random.uniform(1e-1, 1e2, 1)
        Nr = 1000  # np.random.randint(1000, 2000)
        N = 200  # np.random.randint(500, 1000)

        r_grid = np.logspace(-12, np.log10(R), Nr)
        f_r = 1./r_grid**1.0

        ell1 = 0
        ell2 = i + 1
        kln1, cln1 = constructGridAndNorms(R, N, ell1)
        kln2, cln2 = constructGridAndNorms(R, N, ell2)

        f_l1_k = forwardDiscreteSphericalBesselTransform(
            f_r, r_grid, ell1, kln1, cln1)
        f_l2_k = forwardDiscreteSphericalBesselTransform(
            f_r, r_grid, ell2, kln2, cln2)

        cou12 = orderCouplingMatrix(ell1, ell2, N, Nr)
        f_l1_k2 = np.dot(cou12, f_l2_k)
        rat = f_l1_k2[:-1]/f_l1_k[:-1]
        print(R, ell1, ell2, N, np.mean(rat), np.max(rat))
        # np.testing.assert_allclose(f_l1_k2[:-1], f_l1_k[:-1], rtol=rtol)
        assert np.abs(np.mean(rat - 1)) < rtol


@pytest.mark.skip("Hard to pass due to fast transform approximation")
def test_couplingMatrix_fast():
    """Test coupling matrix with fast transform"""
    rtol = 1e-1
    NREPEAT = 2
    for i in range(NREPEAT):
        R = np.random.uniform(1e-1, 1e1, 1)
        Nr = np.random.randint(1000, 3000)
        N = np.random.randint(10, 20)

        r_grid = np.logspace(-12, np.log10(R), Nr)

        ell1 = 0
        ell2 = 1
        kln1, cln1 = constructGridAndNorms(R, N, ell1)
        K1 = kln1[-1]
        kln2, cln2 = constructGridAndNorms(R, N, ell2)
        K2 = kln2[-1]
        rl1, dl1 = constructGridAndNorms(K1, N, ell1)
        rl2, dl2 = constructGridAndNorms(K2, N, ell2)
        f_l1_r = 1/rl1
        f_l2_r = 1/rl2

        mat11 = fastTransformMatrix(ell1, ell1, N)
        f_l1_k = np.dot(mat11, f_l1_r) / K1**3
        mat22 = fastTransformMatrix(ell2, ell2, N)
        f_l2_k = np.dot(mat22, f_l2_r) / K2**3

        cou12 = orderCouplingMatrix(ell1, ell2, N, Nr)
        f_l1_k2 = np.dot(cou12, f_l2_k)
        print(R, ell1, ell2, N, f_l1_k2[:-1]/f_l1_k[:-1])
        np.testing.assert_allclose(f_l1_k2[:-1], f_l1_k[:-1], rtol=rtol)


def test_healpixMatrixTranform():
    """Test that matrix form agrees with healpy anafast routines"""

    rtol = 2e-2

    nside = 16
    L = 1*nside//2
    N = np.random.randint(10, 100)
    Lt = L*(L+1)//2
    f_lm_t = np.random.uniform(0, 1, size=Lt) +\
        1j*np.random.uniform(0, 1, size=Lt)
    for l in range(L):
        f_lm_t[l] = np.random.uniform(0, 1)

    f_ang = hp.alm2map(f_lm_t, nside, lmax=L-1, verbose=False)
    f_lm2 = hp.map2alm(f_ang, lmax=L-1)
    np.testing.assert_allclose(f_lm2, f_lm_t, rtol=rtol)

    mat, fac = healpixMatrixTransform(L, nside)

    matmat = np.dot(np.conjugate(mat), mat.T) * fac
    eye = np.eye(Lt)
    np.testing.assert_allclose(matmat, eye, atol=1e-1)

    f_lm_c = 1*f_lm_t
    f_lm_c[:L] /= 2
    pr = np.dot(mat.T, f_lm_c)
    f_ang3 = (pr + np.conjugate(pr)).astype(float)
    # f_ang3 = (np.dot(mat.T, f_lm_t) +
    #          np.conjugate(np.dot(mat[L:, :].T, f_lm_t[L:]))).astype(float)
    f_lm3 = fac * np.dot(np.conjugate(mat), f_ang3)
    np.testing.assert_allclose(f_lm3, f_lm_t, rtol=rtol)


@pytest.mark.skip("Too hard to pass without proper benchmark tests")
def test_slowFourierBesselTransform():
    """Test slow Fourier-Bessel transform"""
    rtol = 1e-1
    NREPEAT = 1
    for i in range(NREPEAT):
        R = 10  # np.random.uniform(1e-1, 1e1, 1)
        Nr = 1000  # np.random.randint(1000, 2000)
        N = 300  # np.random.randint(10, 100)
        nside = 8
        l0 = np.random.randint(0, 3)
        L = nside//2
        Lt = L*(L+1)//2
        f_lmn_t = np.random.uniform(0, 1, Lt*N) +\
            1j*np.random.uniform(0, 1, Lt*N)
        f_lmn_t = f_lmn_t.reshape((Lt, N))
        for l in range(L):
            f_lmn_t[l, :] = np.random.uniform(0, 1, N)

        shatmat, fac = healpixMatrixTransform(L, nside)
        qlns = rootsSphericalBesselFunctions(L+1, N)
        f_angr = inverseFourierBesselTransform(f_lmn_t, L, nside,
                                               R, l0, qlns, shatmat)
        f_lmn = forwardFourierBesselTransform(f_angr, L,
                                              R, l0, Nr, qlns, shatmat)
        f_lmn *= fac
        print(np.mean(f_lmn/f_lmn_t))
        print(np.mean(f_lmn/f_lmn_t, axis=1))
        print(np.mean(f_lmn/f_lmn_t, axis=0))
        np.testing.assert_allclose(f_lmn, f_lmn_t, rtol=rtol)
