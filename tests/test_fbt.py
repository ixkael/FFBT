
import numpy as np
import pytest

from ffbt.sht import *
from ffbt.sbt import *
from ffbt.fbt import *

def test_fourierBesselMatrix():
    """Test slow Fourier-Bessel transform"""
    rtol = 2e-1
    NREPEAT = 1
    for i in range(NREPEAT):

        R = np.random.uniform(1e-1, 1e2, 1)

        N = 4
        Nr = 1000
        nside = 8
        L = nside // 4

        Npix = hp.nside2npix(nside)
        Lt = L*(L+1)//2

        thetas, phis = hp.pix2ang(nside, np.arange(Npix))
        #rs_grid_bounds = np.linspace(0, R, Nr)
        rs_grid_bounds = np.logspace(1e-6, np.log10(R), Nr)
        rs_grid_mid = (rs_grid_bounds[1:] + rs_grid_bounds[:-1])/2.

        clnmat, sbtmat, shamat, domega, dr = fourierBesselTransformMatrices(L, N, R, nside, rs_grid_bounds)

        f_lmn_t = np.random.rand(Lt, N) + 1j*np.random.rand(Lt, N)
        for l in range(L):
            f_lmn_t[l, :] = np.random.uniform(0, 1, N)
        # f_lmn is (Lt, N)
        # shamat is (Lt, N, Npix)
        # clnmat and sbtmat are (Lt, N, Nr)
        f_angr = forwardFourierBesselTransformMatrix(f_lmn_t, clnmat, sbtmat, shamat)
        # f_angr is (Npix, Nr)
        f_lmn = inverseFourierBesselTransformMatrix(f_angr, clnmat, sbtmat, shamat, domega, dr, rs_grid_mid)
        # f_lmn is (Lt, N)

        #for l in range(L):
        #    for m in range(l+1):
        #        lm = m*(2*L-1-m)//2+l
        #        for n in range(N):
        #            if np.abs(f_lmn[lm, n] - f_lmn_t[lm, n]) > 1e-5:
        #                print(l, m, n, f_lmn[lm, n], f_lmn_t[lm, n])

        np.testing.assert_allclose(f_lmn, f_lmn_t, rtol=rtol, atol=1e-12)


@pytest.mark.skip("Too hard to pass without proper benchmark tests")
def test_slowFourierBesselTransform():
    """Test slow Fourier-Bessel transform"""
    rtol = 1e-1
    NREPEAT = 1
    for i in range(NREPEAT):
        R = 10  # np.random.uniform(1e-1, 1e1, 1)
        Nr = 100  # np.random.randint(1000, 2000)
        N = 100  # np.random.randint(10, 100)
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

        mat = fourierBesselTransformMatrix(L, nside, R, l0, qlns, shatmat, fac)
        f_lmn2 = np.sum(mat[:, :, :, :] * f_angr[None, None, :, :], axis=(2, 3))
        f_angr2 = np.sum(mat[:, :, :, :].T * f_lmn2[None, None, :, :], axis=(2, 3))
        print(np.mean(f_lmn2/f_lmn_t))
        print(np.mean(f_angr2/f_angr, axis=1))
        np.testing.assert_allclose(f_lmn2, f_lmn_t, rtol=rtol)
