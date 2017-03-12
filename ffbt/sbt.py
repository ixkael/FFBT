# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import spherical_jn


def continuousSphericalBesselTransform(f_r, r_finegrid, k_nodes, ell):
    kr = r_finegrid[:, None] * k_nodes[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    r2 = r_finegrid[:, None]**2.
    flk = np.trapz(f_r[:, None] * j_ell_k_r * r2, x=r_finegrid, axis=0)
    return np.sqrt(2/np.pi) * flk


def constructGridAndNorms(R, N):

    return kln_nodes, cln_norms


def forwardDiscreteSphericalBesselTransform(f_r, r_grid, ell, N):
    R = r_grid[-1]
    # TODO: add other boundary conditions
    kln_nodes, cln_norms = constructGridAndNorms(R, N)
    f_l_kln = continuousSphericalBesselTransform(f_r, r_grid, kln_nodes, ell)
    return f_l_kln, kln_nodes, cln_norms


def inverseDiscreteSphericalBesselTransform(f_l_kln, klns, clns, r_grid, ell):
    kr = r_grid[:, None] * klns[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    f_r = np.sum(f_l_kln[None, :] * clns[None, :] * j_ell_k_r, axis=1)
    return f_r
