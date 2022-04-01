import numpy as np
from numba import jit

from turbo_seti.find_doppler.kernels._bitrev import bitrev


@jit(nopython=True)
def flt(outbuf, nchn):
    """
    This is a function to Taylor-tree-sum a data stream. It assumes that
    the arrangement of data stream is, all points in first spectra, all
    points in second spectra, etc. Data are summed across time.

    Parameters
    ----------
    outbuf : array_like
        Input data array, replaced by dedispersed data at the output.
    nchn : int
        Number of timesteps in the data.

    References
    ----------
    - R. Ramachandran, 07-Nov-97, nfra. -- Original algorithm.
    - A. Siemion, 2011 -- float/64 bit addressing (C-code)
    - H. Chen, 2014 -- python version
    - E. Enriquez + P.Schellart, 2016 -- cython version
    - L. Cruz, 2020 -- numba version

    """
    mlen = len(outbuf)
    nsamp = (mlen / nchn) - (2 * nchn)
    npts = nsamp + nchn
    nstages = int(np.log2(nchn))
    ndat1 = nsamp + 2 * nchn
    nmem = 1

    for istages in range(0, nstages):
        nmem *= 2
        nsec1 = int(nchn / nmem)
        nmem2 = nmem - 2

        for isec in range(0, nsec1):
            ndelay = -1
            koff = isec * nmem
            for ipair in range(0, nmem2 + 1, 2):
                ioff1 = int((bitrev(ipair, istages + 1) + koff) * ndat1)
                i2 = int((bitrev(ipair + 1, istages + 1) + koff) * ndat1)
                ndelay += 1
                ndelay2 = ndelay + 1
                nfin = int(npts + ioff1)

                for i1 in range(ioff1, nfin):
                    itemp = outbuf[i1] + outbuf[i2 + ndelay]
                    outbuf[i2] = outbuf[i1] + outbuf[i2 + ndelay2]
                    outbuf[i1] = itemp
                    i2 += 1

    return outbuf
