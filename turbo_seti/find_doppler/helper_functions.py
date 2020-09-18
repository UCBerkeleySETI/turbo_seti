#!/usr/bin/env python

import numpy as np
import logging
logger_hf = logging.getLogger(__name__)


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    """ Apply barycentric frequency correction

    Args:
      header: header dict with 'NAXIS1', 'DELTAF', 'FCNTR' and 'baryv'
      fine_channel: index of fine channel to lookup frequency for
      tdwidth: normally just fft_len
      ref_frame (0 or 1): Barycentric (1) or Topocentric (0) frame
      fscrunch (int): if searching scrunched frequency channels, pass this value

    Returns:

    """
    fftlen = header['NAXIS1']
    deltaf = header['DELTAF']

    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header['FCNTR'] + (chan_index - fftlen/2)*deltaf
    #/* apply doppler correction */
    if ref_frame == 1:
        chanfreq = (1 - header['baryv']) * chanfreq
    return chanfreq


def bitrev(inval, nbits):
    """This function bit-reverses the given value "inval" with the number of
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
        ibitr = 0 if (1 & k == 0) else ifact
        for i in range(2, nbits+1):
            k = k >> 1
            ifact = ifact >> 1
            if 1 & k:
                ibitr += ifact
    return ibitr


def FlipX(outbuf, xdim, ydim):
    """This function takes in an array of values and iteratively flips ydim chunks of values of length xdim. For example,
    if you have an array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and enter it with xdim = 5 and ydim = 2, the array will be
    modified to become [5, 4, 3, 2, 1, 10, 9, 8, 7, 6]. Note that if you wish for the whole array to be modified in this
    way, xdim * ydim should equal the length of the array. If ydim * xdim is greater than the length of the array, this
    function will error.

    Args:
      outbuf: ndarray,        an array with shape = (int, 1)
      xdim: int,            size of segments to be flipped
      ydim: int,            amount of segments of size xdim to be flipped

    Returns:

    """
    np.copyto(outbuf, outbuf.reshape((ydim, xdim))[:, ::-1].ravel())


def comp_stats(np_arr):
    """Compute mean and stddev of floating point vector array in a fast way, without using the outliers.

    Args:
      np_arr: ndarray,        floating point vector array

    Returns:
      the_median, the_stddev : float, float,   median and standard deviation of input array

    """

    new_vec = np.sort(np_arr)

    #Removing the lowest 5% and highest 5% of data, this takes care of outliers.
    new_vec = new_vec[int(len(new_vec)*.05):int(len(new_vec)*.95)]
    the_median = np.median(new_vec)
    the_stddev = new_vec.std()

    return the_median, the_stddev
