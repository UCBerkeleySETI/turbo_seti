#!/usr/bin/env python

import numpy as np
import logging
logger_hf = logging.getLogger(__name__)


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    fftlen = header['NAXIS1']
    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header['FCNTR'] + (chan_index - fftlen/2)*header['DELTAF']
    #/* apply doppler correction */
    if ref_frame == 1:
        chanfreq = (1 - header['baryv']) * chanfreq
    return chanfreq


def bitrev(inval, nbits):
    """
    This function bit-reverses the given value "inval" with the number of
    bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.
    python version ----  H. Chen   Modified 2014
    :param inval:   number to be bit-reversed
    :param nbits:   The length of inval in bits. If user only wants the bit-reverse of a certain amount of bits of
                    inval, nbits is the amount of bits to be reversed counting from the least significant (rightmost)
                    bit. Any bits beyond this length will not be reversed and will be truncated from the result.
    :return:        the bit-reverse of inval. If there are more significant bits beyond nbits, they are truncated.
    """
    if nbits <= 1:
        ibitr = inval
    else:
        ifact = 1
        for i in range(1, nbits):
           ifact *= 2
        k = inval
        ibitr = (1 & k) * ifact
        for i in range(2, nbits+1):
            k = int(k / 2)
            ifact = int(ifact / 2)
            ibitr += (1 & k) * ifact
    return ibitr


def FlipX(outbuf, xdim, ydim):
    temp = np.empty_like(outbuf[0:xdim])
    logger_hf.debug("FlipX: temp array dimension: %s"%str(temp.shape))

    for j in range(0, ydim):
        indx = j * xdim
        np.copyto(temp, outbuf[indx:indx+xdim])
        np.copyto(outbuf[indx: indx+xdim], temp[::-1])
    return


def comp_stats(arrey):
    #Compute mean and stddev of floating point vector array in a fast way, without using the outliers.

    new_vec = np.sort(arrey,axis=None)

    #Removing the lowest 5% and highest 5% of data, this takes care of outliers.
    new_vec = new_vec[int(len(new_vec)*.05):int(len(new_vec)*.95)]
    the_median = np.median(new_vec)
    the_stddev = new_vec.std()

    return the_median, the_stddev
