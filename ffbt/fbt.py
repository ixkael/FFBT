# -*- coding: utf-8 -*-

import numpy as np

from ffbt.sht import *
from ffbt.sbt import *


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


def fourierBesselTransformMatrices(L, N, R, nside, rs_grid_bounds):
    Npix = 12*int(nside)**2
    Lt = L*(L+1)//2
    shamat = np.zeros((Lt, Npix), dtype=complex)
    thetas, phis = hp.pix2ang(nside, np.arange(Npix))
    for l in range(L):
        for m in range(l+1):
            lm = m*(2*L-1-m)//2+l
            shamat[lm, :] = sph_harm(m, l, phis, thetas)
    domega = 4 * np.pi / Npix

    rs_grid_mid = (rs_grid_bounds[1:] + rs_grid_bounds[:-1])/2.
    sbtmat = np.zeros((Lt, N, rs_grid_mid.size))
    clnmat = np.zeros((Lt, N))
    dr = (rs_grid_bounds[1:] - rs_grid_bounds[:-1])
    for l in range(L):
        kln_nodes, cln_norms = constructGridAndNorms(R, N, l)
        kr = rs_grid_mid[None, :] * kln_nodes[:, None]
        j_ell_k_r = spherical_jn(l, kr, derivative=False)
        for m in range(l+1):
            lm = m*(2*L-1-m)//2+l
            sbtmat[lm, :, :] = j_ell_k_r
            clnmat[lm, :] = cln_norms
    return clnmat, sbtmat, shamat, domega, dr


def fourierBesselTransformMatrix(L, N, R, phis, thetas, rs):
    nobj = rs.size
    assert rs.size == phis.size
    assert rs.size == thetas.size
    Lt = L*(L+1)//2
    mat = np.zeros((Lt*N, nobj), dtype=complex)
    for ell in range(L):
        kln_nodes, cln_norms = constructGridAndNorms(R, N, ell)
        kr = rs[:, None] * kln_nodes[None, :]
        j_ell_k_r = spherical_jn(ell, kr, derivative=False)
        for em in range(ell+1):
            sha = sph_harm(em, ell, phis, thetas)
            for en in range(N):
                lmn = en*Lt + ell*(ell+1)//2 + em
                mat[lmn, :] = sha[:] * j_ell_k_r[:, en]
    return mat


def fourierBesselTransformMatrix_approx(L, nside, R, l0, qlns, shamat, Nr, fac):
    # f_angr = np.zeros((Npix, N))
    # f_lmn = np.zeros((Lt, N), dtype=complex)
    # shamat = np.zeros((Lt, Npix), dtype=complex)
    #mat = np.zeros((Lt, N, Npix, N), dtype=complex)
    N = qlns.shape[1]
    klns, clns = constructGridAndNorms(R, N, l0)
    K = klns[-1]
    Npix = 12*nside*nside
    Lt = L*(L+1)//2
    sbtmat = fastTransformMatrix(l0, l0, N, qlns)
    sfbmat = shamat[:, None, :, None] * np.ones((Lt, N, Npix, N))
    for l in range(L):
        cou = orderCouplingMatrix(l, l0, N, Nr, qlns)
        for m in range(l+1):
            i = m*(2*L-1-m)//2+l
            sfbmat[i, :, :, :] *= np.dot(cou, sbtmat)[:, None, :] / K**3
    return sfbmat * fac

# fastForwardFourierBesselTransform
# fastInverseFourierBesselTransform
