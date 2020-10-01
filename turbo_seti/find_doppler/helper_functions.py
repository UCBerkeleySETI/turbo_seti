#!/usr/bin/env python

import logging
logger_hf = logging.getLogger(__name__)


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    """

    Args:
      header: 
      fine_channel: 
      tdwidth: 
      ref_frame: 

    Returns:

    """
    fftlen = header['NAXIS1']
    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header['FCNTR'] + (chan_index - fftlen/2)*header['DELTAF']
    #/* apply doppler correction */
    if ref_frame == 1:
        chanfreq = (1 - header['baryv']) * chanfreq
    return chanfreq


def FlipX(xp, outbuf, xdim, ydim):
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
    xp.copyto(outbuf, outbuf.reshape((ydim, xdim))[:, ::-1].ravel())


def comp_stats(xp, np_arr):
    """Compute mean and stddev of floating point vector array in a fast way, without using the outliers.

    Args:
      np_arr: ndarray,        floating point vector array

    Returns:
      the_median, the_stddev : float, float,   median and standard deviation of input array

    """

    new_vec = xp.sort(np_arr)

    #Removing the lowest 5% and highest 5% of data, this takes care of outliers.
    new_vec = new_vec[int(len(new_vec)*.05):int(len(new_vec)*.95)]
    the_median = xp.median(new_vec)
    the_stddev = new_vec.std()

    return the_median, the_stddev
