#!/usr/bin/env python

import numpy as np


def chan_freq(header, fine_channel, tdwidth, ref_frame):
    fftlen = header['NAXIS1']
    chan_index = fine_channel - (tdwidth-fftlen)/2
    chanfreq = header['FCNTR'] + (chan_index - fftlen/2)*header['DELTAF']
    #/* apply doppler correction */
    if ref_frame == 1:
        chanfreq = (1 - header['baryv']) * chanfreq
    return chanfreq

