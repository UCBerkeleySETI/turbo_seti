#!/usr/bin/env python

import numpy as np
import astropy.io.fits as pyfits
try:
    from .helper_functions import chan_freq
except:
    from helper_functions import chan_freq

import logging

def tophits_writer(spectra_slice, hit_indices, header, format='txt'):
    return None

class GeneralWriter:
    """ """
    def __init__(self, filename='', mode='a'):
        with open(filename, mode) as myfile:
            self.filehandle = myfile
            self.filename = filename
        return None

    def close(self):
        if self.filehandle.closed:
            pass
        else:
            self.filehandle.close()

    def open(self, mode='a'):
        if self.filehandle.closed:
            with open(self.filename, mode) as myfile:
                self.filehandle = myfile
        elif self.filehandle.mode == mode:
            return
        else:
            self.close()
            with open(self.filename, mode) as myfile:
                self.filehandle = myfile

    def is_open(self):
        return not self.filehandle.closed

    def writable(self):
        if self.is_open() and (('w' in self.filehandle.mode) or ('a' in self.filehandle.mode)):
            return True
        else:
            return False

    def write(self, info_str, mode='a'):
        if (not 'w' in mode) and (not 'a' in mode):
            mode = 'a'
        if not self.writable():
            with open(self.filename, mode) as myfile:
                myfile.write(info_str)
                self.filehandle = myfile
        else:
            self.filehandle.write(info_str)

    def start_over(self):
        self.open('w')
        self.write('')
        self.open('a')

class FileWriter(GeneralWriter):
    """ """
    def __init__(self, filename, header):
        GeneralWriter.__init__(self, filename)
        self.write('# -------------------------- o --------------------------\n')
        self.write('# File ID: %s \n'%(filename.split('/')[-1].replace('.dat','')+'.h5'))
        self.write('# -------------------------- o --------------------------\n')
        self.report_header(header)

        self.tophit_count = 0

    def report_coarse_channel(self, header,total_n_candi):
        ''' Write header information per given obs.
        '''

        return None
        self.write('# Coarse Channel Number: %i \n'%header[u'coarse_chan'])
        info_str = '# Number of hits: %i \n'%total_n_candi
        self.write(info_str)

    def report_header(self, header):
        ''' Write header information per given obs.
        '''

        info_str = '# Source:%s\n# MJD: %18.12f\tRA: %s\tDEC: %s\n# DELTAT: %10.6f\tDELTAF(Hz): %10.6f\n'%(header[u'SOURCE'],header[u'MJD'], header[u'RA'], header[u'DEC'], header[u'DELTAT'], header[u'DELTAF']*1e6)

        self.write(info_str)
        self.write('# --------------------------\n')
        info_str = '# Top_Hit_# \t'
        info_str += 'Drift_Rate \t'
        info_str += 'SNR \t'
        info_str += 'Uncorrected_Frequency \t'
        info_str += 'Corrected_Frequency \t'
        info_str += 'Index \t'
        info_str += 'freq_start \t'
        info_str += 'freq_end \t'
        info_str += 'SEFD \t'
        info_str += 'SEFD_freq \t'
        info_str += 'Coarse_Channel_Number \t'
        info_str += 'Full_number_of_hits \t'
        info_str +='\n'
        self.write(info_str)
        self.write('# --------------------------\n')

    def report_tophit(self, max_val, ind, ind_tuple, tdwidth, fftlen, header,total_n_candi,spec_slice=None,obs_info=None):

        '''This function looks into the top hit in a region, basically find the local maximum and saves that.
        '''

        offset = int((tdwidth - fftlen)/2)
        tdwidth =  len(max_val.maxsnr)

        self.tophit_count += 1
        freq_start = chan_freq(header, ind_tuple[0]-offset, tdwidth, 0)
        freq_end = chan_freq(header, ind_tuple[1]-1-offset, tdwidth, 0)

        uncorr_freq = chan_freq(header, ind-offset, tdwidth, 0)
        corr_freq = chan_freq(header, ind-offset, tdwidth, 1)

        #Choosing the index of given SEFD and freq.
        if obs_info['SEFDs_freq'][0] > 0.:
            this_one = np.arange(len(obs_info['SEFDs_freq']))[ (obs_info['SEFDs_freq_up']>uncorr_freq) ][0]
        else:
            this_one = 0

        info_str = '%03d\t'%(self.tophit_count)  #Top Hit number
        info_str += '%10.6f\t'%max_val.maxdrift[ind]  #Drift Rate
        info_str += '%10.6f\t'%max_val.maxsnr[ind]  #SNR
        info_str += '%14.6f\t'%uncorr_freq #Uncorrected Frequency:
        info_str += '%14.6f\t'%corr_freq #Corrected Frequency:
        info_str += '%d\t'%(ind - offset) #Index:
        info_str += '%14.6f\t'%freq_start #freq_start:
        info_str += '%14.6f\t'%freq_end #freq_end:
        info_str += '%s\t'%obs_info['SEFDs_val'][this_one] #SEFD:
        info_str += '%14.6f\t'%obs_info['SEFDs_freq'][this_one] #SEFD_mid_freq:
        info_str += '%i\t'%header[u'coarse_chan'] #
        info_str += '%i\t'%total_n_candi #
        info_str +='\n'
        self.write(info_str)

        return self

class LogWriter(GeneralWriter):
    """ """
    def report_candidate(self, info_str):
        return None

    def info(self, info_str):
        self.write(info_str + '\n')
        return None



