#!/usr/bin/env python
"""This is a function to Taylor-tree-sum a data stream. It assumes that
the arrangement of data stream is, all points in first spectra, all
points in second spectra, etc...  Data are summed across time.
         Original algorithm: R. Ramachandran, 07-Nov-97, nfra.
         Modified 2011 A. Siemion float/64 bit addressing (C-code)
         Modified 2014 H. Chen python version
         Modified 1-Feb-2016 E. Enriquez + P.Schellart cython version
"""

import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def taylor_flt(np.ndarray[DTYPE_t, ndim=1] outbuf, long mlen, long nchn):
    """
    Parameters:
        outbuf       : input array (float), replaced by dedispersed data
                       at the output
        mlen         : dimension of outbuf[] (long int)
        nchn         : number of frequency channels (long int)
    """

    assert outbuf.dtype == DTYPE

    cdef long nsamp = (mlen/nchn) - (2*nchn)
    cdef long npts = nsamp + nchn
    cdef int nstages = int(np.log2(nchn))

    cdef int istages
    cdef int isec
    cdef int ipair
    cdef int i1
    cdef int ndelay
    cdef int ndelay2
    cdef int nmem = 1
    cdef int nmem2
    cdef int nsec1

    cdef int ndat1 = nsamp + 2 * nchn
    cdef int koff
    cdef int nfin
    cdef int ioff1
    cdef int i2

    cdef DTYPE_t itemp

    for istages in range(0, nstages):
        nmem  *= 2
        nsec1  = nchn/nmem
        nmem2  = nmem - 2
        for isec in range(0, nsec1):
            ndelay = -1
            koff = isec * nmem
            for ipair in range(0, nmem2+1, 2):
                ioff1 = (bitrev(ipair, istages+1) + koff) * ndat1
                i2 = (bitrev(ipair+1, istages+1) + koff) * ndat1
                ndelay += 1
                ndelay2 = (ndelay + 1)
                nfin = (npts + ioff1)

                for i1 in range(ioff1, nfin):
                    itemp = outbuf[i1] + outbuf[i2+ndelay]
                    outbuf[i2] = outbuf[i1] + outbuf[i2+ndelay2]
                    outbuf[i1] = itemp
                    i2 += 1
    return

"""This function bit-reverses the given value "inval" with the number of
bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.
python version ----  H. Chen   Modified 2014
cython version ---- E. Enriquez + P.Schellart 1-Feb-2016
"""
cpdef int bitrev(int inval, int nbits) except *:
    cdef int ibitr
    cdef int k
    cdef int ifact

    if nbits <= 1:
        ibitr = inval
    else:
        ifact = 1
        for i in range(1, nbits):
           ifact *= 2
        k = inval
        ibitr = (1 & k) * ifact
        for i in range(2, nbits+1):
            k /= 2
            ifact /= 2
            ibitr += (1 & k) * ifact
    return ibitr

