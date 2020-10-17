import cupy as cp
import numpy as np

def flt(outbuf, mlen, nchn):
    r"""
    This is a function to Taylor-tree-sum a data stream. It assumes that
    the arrangement of data stream is, all points in first spectra, all
    points in second spectra, etc. Data are summed across time.
 
    Parameters
    ----------
    outbuf : array_like
        Input data array, replaced by dedispersed data at the output.
    mlen : int
        Dimension of outbuf[].
    nchn : int
        Number of frequency channels.

    References
    ----------
    - R. Ramachandran, 07-Nov-97, nfra. -- Original algorithm.
    - A. Siemion, 2011 -- float/64 bit addressing (C-code)
    - H. Chen, 2014 -- python version
    - E. Enriquez + P.Schellart, 2016 -- cython version
    - L. Cruz, 2020 -- vectorized version

    """
    nsamp = (mlen/nchn) - (2*nchn)
    npts = nsamp + nchn
    nstages = int(np.log2(nchn))
    ndat1 = nsamp + 2 * nchn
    nmem = 1

    for istages in range(0, nstages):
        nmem  *= 2
        nsec1  = nchn//nmem
        nmem2  = nmem - 2
        
        for isec in range(0, nsec1):
            ndelay = -1
            koff = isec * nmem
            for ipair in range(0, nmem2+1, 2):
                ioff1 = int((bitrev(ipair, istages+1) + koff) * ndat1)
                i2 = int((bitrev(ipair+1, istages+1) + koff) * ndat1)
                ndelay += 1
                ndelay2 = (ndelay + 1)
                nfin = int(npts + ioff1)

                l1 = (nfin - ioff1)
                a = outbuf[ioff1:nfin]
                b = outbuf[i2+ndelay:i2+ndelay+l1]
                c = outbuf[i2+ndelay2:i2+ndelay2+l1]

                outbuf[ioff1:nfin], outbuf[i2:i2+l1] = sum(a, b, c)


@cp.fuse()
def sum(a, b, c):
    """
    """
    return a + b, a + c


def bitrev(inval, nbits):
    r"""
    This function bit-reverses the given value "inval" with the number of bits, "nbits".
    
    Parameters
    ----------
    inval : int
      Number to be bit-reversed.
    nbits : int
      The length of inval in bits. If user only wants the bit-reverse of a certain amount of bits of
      inval, nbits is the amount of bits to be reversed counting from the least significant (rightmost)
      bit. Any bits beyond this length will not be reversed and will be truncated from the result.
    
    Returns
    -------
    : int
      The bit-reverse of inval. If there are more significant bits beyond nbits, they are truncated.

    References
    ----------
    - R. Ramachandran, 10-Nov-97, nfra. -- Original C implementation.
    - H. Chen, 2014 -- Python version.
    - R. Elkins (texadactyl), 2020 -- Speedup.

    """
    if nbits <= 1:
        ibitr = inval
    else:
        ifact = 2**(nbits - 1)
        k = inval
        ibitr = (1 & k) * ifact
        for _ in range(2, nbits+1):
            k = k >> 1
            ifact = ifact >> 1
            if 1 & k:
                ibitr += ifact
    return ibitr