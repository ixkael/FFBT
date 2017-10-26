
import numpy as np
from numpy.polynomial.polynomial import polyval
import pytest
from scipy.special import gammaln
from scipy.special import spherical_jn
from time import time

from ffbt.sht import *


def test_healpixMatrixTranform():
    """Test that matrix form agrees with healpy anafast routines"""

    rtol = 1e-1

    nside = 32
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

    t1 = time()
    mat, fac = healpixMatrixTransform(L, nside)
    t2 = time()
    mat2, fac2 = healpixMatrixTransform2(L, nside)
    t3 = time()
    np.testing.assert_allclose(mat, mat2, atol=1e-12)
    print(t2-t1, t3-t2)

    matmat = np.dot(np.conjugate(mat), mat.T) * fac
    eye = np.eye(Lt)
    np.testing.assert_allclose(matmat, eye, atol=1e-1)

    f_lm_c = 1*f_lm_t
    f_lm_c[:L] /= 2
    pr = np.dot(mat.T, f_lm_c)
    f_ang3 = (pr + np.conjugate(pr)).real
    # f_ang3 = (np.dot(mat.T, f_lm_t) +
    #          np.conjugate(np.dot(mat[L:, :].T, f_lm_t[L:]))).astype(float)
    f_lm3 = fac * np.dot(np.conjugate(mat), f_ang3)
    np.testing.assert_allclose(f_lm3, f_lm_t, rtol=rtol)
