#!/usr/bin/env python

import logging
import numpy as np
logger_hf = logging.getLogger(__name__)


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    r"""
    Find channel frequency.

    Parameters
    ----------
      header : 
      fine_channel : 
      tdwidth : 
      ref_frame : 

    Returns
    -------
    chanfreq : float

    """
    fftlen = header['NAXIS1']
    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header['FCNTR'] + (chan_index - fftlen/2)*header['DELTAF']
    # apply doppler correction
    if ref_frame == 1:
        chanfreq = (1 - header['baryv']) * chanfreq
    return chanfreq


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
        ibitr = 0 if (1 & k == 0) else ifact
        for _ in range(2, nbits+1):
            k = k >> 1
            ifact = ifact >> 1
            if 1 & k:
                ibitr += ifact
    return ibitr


def FlipX(outbuf, xdim, ydim, xp=None):
    r"""
    This function takes in an array of values and iteratively flips ydim chunks of values of length xdim.

    Parameters
    ----------
    outbuf : ndarray
      An array with shape like (int, 1)
    xdim : int
      Size of segments to be flipped.
    ydim : int
      Amount of segments of size xdim to be flipped.
    xp : Numpy or Cupy, optional
      Math module to be used. If `None`, Numpy will be used.

    Examples
    --------
    If you have an array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and enter it with xdim = 5 and ydim = 2, the array will be
    modified to become [5, 4, 3, 2, 1, 10, 9, 8, 7, 6]. Note that if you wish for the whole array to be modified in this
    way, xdim * ydim should equal the length of the array. If ydim * xdim is greater than the length of the array, this
    function will error.

    """
    if not xp:
      xp = np

    xp.copyto(outbuf, outbuf.reshape((ydim, xdim))[:, ::-1].ravel())


def comp_stats(np_arr, xp=None):
    r"""
    Compute mean and stddev of floating point vector array in a fast way, without using the outliers.

    Parameters
    ----------
    np_arr : ndarray
      Floating point vector array.
    xp : Numpy or Cupy, optional
      Math module to be used. If `None`, Numpy will be used.

    Returns
    -------
    the_median, the_stddev : float, float
      Median and standard deviation of input array.

    """
    if not xp:
      xp = np

    new_vec = xp.sort(np_arr)

    #Removing the lowest 5% and highest 5% of data, this takes care of outliers.
    new_vec = new_vec[int(len(new_vec)*.05):int(len(new_vec)*.95)]
    the_median = xp.median(new_vec)
    the_stddev = new_vec.std()

    return the_median, the_stddev
