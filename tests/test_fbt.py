
import numpy as np
import pytest

from ffbt.fbt import *


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
