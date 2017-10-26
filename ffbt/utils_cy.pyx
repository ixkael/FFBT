#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow
from scipy.special.cython_special cimport sph_harm


# TODO: directly compute and store all legendre poly, then mult by complex exponential

def spherharmgrid(
            double complex [:, :] mat, # L*(L+1)//2, Npix
            long L,
            long Npix,
            double[:] thetas, #Npix
            double[:] phis): #Npix
    cdef int Lt = L*(L+1)//2
    cdef int lm
    cdef int l, m, o
    for o in prange(Npix, nogil=True):
        for l in range(L):
            for m in range(l+1):
                lm = m*(2*L-1-m)//2+l
                mat[lm, o] = sph_harm(m, l, phis[o], thetas[o])
