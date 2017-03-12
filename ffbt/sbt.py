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


def forwardDiscreteSphericalBesselTransform(f_r, r_grid, ell, N,
                                            kln_nodes=None, cln_norms=None):
    R = r_grid[-1]
    if kln_nodes is None or cln_norms is None:
        kln_nodes, cln_norms = constructGridAndNorms(R, N, ell)
        f_l_kln = continuousSphericalBesselTransform(f_r, r_grid,
                                                     kln_nodes, ell)
        return f_l_kln, kln_nodes, cln_norms
    else:
        f_l_kln = continuousSphericalBesselTransform(f_r, r_grid,
                                                     kln_nodes, ell)
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
