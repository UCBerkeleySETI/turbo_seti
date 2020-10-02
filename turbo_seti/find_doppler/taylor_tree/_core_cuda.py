import cupy as cp
import numpy as np

def flt(outbuf, mlen, nchn):
    """
    This is a function to Taylor-tree-sum a data stream. It assumes that
    the arrangement of data stream is, all points in first spectra, all
    points in second spectra, etc...  Data are summed across time.
         Original algorithm: R. Ramachandran, 07-Nov-97, nfra.
         Modified 2011 A. Siemion float/64 bit addressing (C-code)
         Modified 2014 H. Chen python version
         Modified 1-Feb-2016 E. Enriquez + P.Schellart cython version

    Args:
        outbuf       : input array (float), replaced by dedispersed data
                       at the output
        mlen         : dimension of outbuf[] (long int)
        nchn         : number of frequency channels (long int)
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
    return a + b, a + c


def bitrev(inval, nbits):
    """
    This function bit-reverses the given value "inval" with the number of
    bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.
    python version ----  H. Chen   Modified 2014
    2020-07-21 speedup --- R. Elkins (texadactyl)

    Args:
      inval: number to be bit-reversed
      nbits: The length of inval in bits. If user only wants the bit-reverse of a certain amount of bits of
    inval, nbits is the amount of bits to be reversed counting from the least significant (rightmost)
    bit. Any bits beyond this length will not be reversed and will be truncated from the result.

    Returns:
      : the bit-reverse of inval. If there are more significant bits beyond nbits, they are truncated.

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