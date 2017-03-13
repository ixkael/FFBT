# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import spherical_jn, jv, jvp
from scipy.optimize import brentq


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


def fastTransformMatrix(ell1, ell2, N):
    L = np.max([ell1, ell2]) + 1
    qlns = rootsSphericalBesselFunctions(L, N)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = np.sqrt(2*np.pi) /\
                spherical_jn(ell1+1, qlns[ell1, j])**2.0\
                * spherical_jn(ell1, qlns[ell2, i] * qlns[ell1, j] /
                               qlns[ell1, -1])
    return mat


def orderCouplingMatrix(ell1, ell2, N, Nr):
    L = np.max([ell1, ell2]) + 1
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
