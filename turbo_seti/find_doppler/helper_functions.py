#!/usr/bin/env python

import numpy as np
import logging
logger_hf = logging.getLogger(__name__)


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    fftlen = header[b'NAXIS1']
    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header[b'FCNTR'] + (chan_index - fftlen/2)*header[b'DELTAF']
    #/* apply doppler correction */
    if ref_frame == 1:
        chanfreq = (1 - header[b'baryv']) * chanfreq
    return chanfreq


#  ======================================================================  #
#  This function bit-reverses the given value "inval" with the number of   #
#  bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.               #
#  python version ----  H. Chen   Modified 2014                            #
#  ======================================================================  #
def bitrev(inval, nbits):
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

#  ======================================================================  #
#  This function bit-reverses the given value "inval" with the number of   #
#  bits, "nbits".                                                          #
#  python version ----  H. Chen   Modified 2014                            #
#  reference: stackoverflow.com/questions/12681945                         #
#  ======================================================================  #
def bitrev2(inval, nbits, width=32):
    b = '{:0{width}b}'.format(inval, width=width)
    ibitr = int(b[-1:(width-1-nbits):-1], 2)
    return ibitr

#  ======================================================================  #
#  This function bit-reverses the given value "inval" with 32bits    #
#  python version ----  E.Enriquez   Modified 2016                            #
#  reference: stackoverflow.com/questions/12681945                         #
#  ======================================================================  #
def bitrev3(x):
    raise DeprecationWarning("WARNING: This needs testing.")

    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16)
    return x

def AxisSwap(inbuf, outbuf, nchans, NTSampInRead):
    #long int    j1, j2, indx, jndx;
    for j1 in range(0, NTSampInRead):
        indx  = j1 * nchans
        for j2 in range(nchans-1, -1, -1):
            jndx = j2 * NTSampInRead + j1
            outbuf[jndx]  = inbuf[indx+j2]

def FlipBand(outbuf, nchans, NTSampInRead):
    temp = np.zeros(nchans*NTSampInRead, dtype=np.float64)

    indx  = (nchans - 1);
    for i in range(0, nchans):
        jndx = (indx - i) * NTSampInRead
        kndx = i * NTSampInRead
        np.copyto(temp[jndx: jndx+NTSampInRead], outbuf[kndx + NTSampInRead])
    #memcpy(outbuf, temp, (sizeof(float)*NTSampInRead * nchans));
    outbuf = temp
    return

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
