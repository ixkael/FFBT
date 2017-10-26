# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import spherical_jn, jv, jvp
from scipy.optimize import brentq
import healpy as hp
from scipy.special import sph_harm, lpmn

from ffbt.utils_cy import spherharmgrid

# TODO: USE or ADD REAL SPHERICAL HARMONICS and compare

def healpixMatrixTransform(L, nside):
    Lt = L*(L+1)//2
    Npix = 12*int(nside)**2
    mat = np.zeros((Lt, Npix), dtype=complex)
    thetas, phis = hp.pix2ang(nside, np.arange(Npix))
    spherharmgrid(mat, L, Npix, thetas, phis)
    fac = 4 * np.pi / Npix
    return mat, fac

def healpixMatrixTransform2(L, nside):
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

def healpixMatrixTransform3(L, nside):
    Lt = L*(L+1)//2
    Npix = 12*int(nside)**2
    mat = np.zeros((Lt, Npix), dtype=complex)
    thetas, phis = hp.pix2ang(nside, np.arange(Npix))

    for o in range(Npix):
        Pmn_z, Pmn_d_z = lpmn(L-1, L-1, np.cos(thetas[o]))
        for l in range(L):
            for m in range(l+1):
                lm = m*(2*L-1-m)//2+l
                fac = (2*l+1) / (4*np.pi) / np.prod(np.arange(l - m + 1, l + m + 1))
                if ~np.isfinite(fac) or fac < 0:
                    fac = 1.
                mat[lm, o] = np.sqrt(fac) * Pmn_z[m, l] * np.exp(1j*m*phis[o])
    fac = 4 * np.pi / Npix
    return mat, fac
