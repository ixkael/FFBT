# -*- coding: utf-8 -*-

import numpy as np


def forwardFourierBesselTransform(f_angr, L, R, l0, Nr, qlns, shamat):
    r_grid = np.logspace(-8, np.log10(R), Nr)
    Lt = L*(L+1)//2
    Npix, N = f_angr.shape
    nside = int(np.sqrt(Npix/12.))
    klns, clns = constructGridAndNorms(R, N, l0)
    K = klns[-1]
    f_lm_r = np.zeros((Lt, N), dtype=complex)
    for i in range(N):
        f_lm_r[:, i] = np.dot(np.conjugate(shamat), f_angr[:, i])
    f_lmn = np.zeros((Lt, N), dtype=complex)
    mat = fastTransformMatrix(l0, l0, N, qlns)
    for l in range(L):
        cou = orderCouplingMatrix(l, l0, N, Nr, qlns)
        for m in range(l+1):
            i = m*(2*L-1-m)//2+l
            tmp = np.dot(mat, f_lm_r[i, :]) / K**3
            f_lmn[i, :] = np.dot(cou, tmp)
    return f_lmn


def inverseFourierBesselTransform(f_lmn, L, nside, R, l0, qlns, shamat):
    Lt = L*(L+1)//2
    Npix = 12*int(nside)**2
    tt, N = f_lmn.shape
    f_lm_r = np.zeros((Lt, N), dtype=complex)
    for l in range(L):
        mat = fastTransformMatrix(l, l0, N, qlns)
        for m in range(l+1):
            i = m*(2*L-1-m)//2+l
            f_lm_r[i, :] = np.dot(mat, f_lmn[i, :]) / R**3
    f_angr = np.zeros((Npix, N))
    f_lm_r[:L, :] /= 2
    for i in range(N):
        pr = np.dot(shamat.T, f_lm_r[:, i])
        f_angr[:, i] = (pr + np.conjugate(pr)).astype(float)

    return f_angr


# fastForwardFourierBesselTransform
# fastInverseFourierBesselTransform
