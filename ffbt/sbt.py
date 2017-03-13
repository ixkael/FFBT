# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import spherical_jn, jv, jvp
from scipy.optimize import brentq
import healpy as hp
from scipy.special import sph_harm


def continuousSphericalBesselTransform(f_r, r_finegrid, k_nodes, ell):
    kr = r_finegrid[:, None] * k_nodes[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    r2 = r_finegrid[:, None]**2.
    flk = np.trapz(f_r[:, None] * j_ell_k_r * r2, x=r_finegrid, axis=0)
    return np.sqrt(2/np.pi) * flk


def constructGridAndNorms(R, N, ell):
    # TODO: add other boundary conditions
    roots = rootsSphericalBesselFunctions(ell+1, N)
    qln = roots[ell, :]
    kln_nodes = qln / R
    cln_norms = np.sqrt(2*np.pi) / R**3 / spherical_jn(ell+1, qln)**2.
    return kln_nodes, cln_norms


def forwardDiscreteSphericalBesselTransform(f_r, r_grid, ell,
                                            kln_nodes, cln_norms):
    R = r_grid[-1]
    f_l_kln = continuousSphericalBesselTransform(f_r, r_grid, kln_nodes, ell)
    return f_l_kln


def inverseDiscreteSphericalBesselTransform(f_l_kln, klns, clns, r_grid, ell):
    kr = r_grid[:, None] * klns[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    f_r = np.sum(f_l_kln[None, :] * clns[None, :] * j_ell_k_r, axis=1)
    return f_r


def rootsSphericalBesselFunctions(L, N):
    ir = np.arange(N+1)
    ranges = (ir + 1/2.)*np.pi - 1./(ir + 1/2.)/np.pi
    roots = np.zeros((L, N))
    for ell in range(L):
        def Jn(r):
            return spherical_jn(ell, r)
        for j in range(N):
            roots[ell, j] = brentq(Jn, ranges[j], ranges[j+1])
        ranges[:-1] = roots[ell, :]
        ranges[-1] = ranges[-2] + ranges[-2] - ranges[-3]
    return roots


def fastTransformMatrix(ell1, ell2, N, qlns=None):
    # TODO: implement in cython
    L = np.max([ell1, ell2]) + 1
    if qlns is None:
        qlns = rootsSphericalBesselFunctions(L, N)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = np.sqrt(2*np.pi) /\
                spherical_jn(ell1+1, qlns[ell1, j])**2.0\
                * spherical_jn(ell1, qlns[ell2, i] * qlns[ell1, j] /
                               qlns[ell1, -1])
    return mat


def orderCouplingMatrix(ell1, ell2, N, Nr, qlns=None):
    # TODO: implement in cython
    L = np.max([ell1, ell2]) + 1
    if qlns is None:
        qlns = rootsSphericalBesselFunctions(L, N)
    qln1 = qlns[ell1, :]
    qln2 = qlns[ell2, :]
    mat = np.zeros((N, N))
    grid = np.linspace(0, 1, Nr)
    for i in range(N):
        for j in range(N):
            f_r = spherical_jn(ell1, grid*qln1[i]) *\
                spherical_jn(ell2, grid*qln2[j])
            val = np.trapz(grid**2 * f_r, x=grid)
            mat[i, j] = val
        mat[i, :] *= 2 / spherical_jn(ell1+1, qln1)**2.
    return mat


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
        f_angr[:, i] = pr + np.conjugate(pr)

    return f_angr


def healpixMatrixTransform(L, nside):
    # TODO: implement in cython
    Lt = L*(L+1)//2
    Npix = 12*int(nside)**2
    mat = np.zeros((Lt, Npix), dtype=complex)
    thetas, phis = hp.pix2ang(nside, np.arange(Npix))
    for l in range(L):
        for m in range(l+1):
            lm = m*(2*L-1-m)//2+l
            mat[lm, :] = sph_harm(m, l, phis, thetas)
    fac = 4 * np.pi / Npix
    return mat, fac


# fastForwardFourierBesselTransform
# fastInverseFourierBesselTransform
