from numba import jit


@jit(nopython=True)
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