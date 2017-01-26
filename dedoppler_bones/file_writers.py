#!/usr/bin/env python

import numpy as np
import astropy.io.fits as pyfits
from helper_functions import chan_freq

import logging
logger = logging.getLogger(__name__)


def tophits_writer(spectra_slice, hit_indices, header, format='txt'):
    return None


def log_writer(filename, info_str):
    return None


def fits_writer(filename, header, fitsdata):
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
    def __init__(self, filename):
        GeneralWriter.__init__(self, filename)
        self.write('File ID: %s \n'%filename.split('/')[-1].replace('.dat',''))
        self.tophit_count = 0

    def report_header(self, header, obs_info=None,LOFAR=False):
        ''' Write header information per given obs.
        '''

        #info_str = 'Source:%s\tMJD: %18.12f\tRA: %10.8f\tDEC: %10.8f\tDELTAT: %10.6f\tDELTAF(Hz): %10.6f\n'%(header['SOURCE'],header['MJD'], header['RA'], header['DEC'], header['DELTAT'], header['DELTAF']*1e6)
        info_str = 'Source:%s\tMJD: %18.12f\tRA: %s\tDEC: %s\tDELTAT: %10.6f\tDELTAF(Hz): %10.6f\n'%(header['SOURCE'],header['MJD'], header['RA'], header['DEC'], header['DELTAT'], header['DELTAF']*1e6)       

        self.write(info_str)
        self.write('--------------------------\n')

        if obs_info and LOFAR:
            info_str= 'RA_tile:%f\tDEC_tile:%f\tRA_beam:%f\tDEC_beam:%f\tRA_TAB:%f\tDEC_TAB:%f\n'%(obs_info['RA_tile'],obs_info['DEC_tile'],obs_info['RA_beam'],obs_info['DEC_beam'],obs_info['RA_TAB'],obs_info['DEC_TAB'])
            info_str+= 'Pulsar_run:%i\tPulsar_found:%i\tPulsar_DM:%f\tPulsar_SNR:%f\n'%(obs_info['pulsar'],obs_info['pulsar_found'],obs_info['pulsar_dm'],obs_info['pulsar_snr'])
            info_str+= 'RFI_level:%f\tN_stations:%i\n'%(obs_info['RFI_level'],obs_info['N_stations'])
            info_str+= 'Mean_SEFD:%s\tpsrflux_Sens:%s\n'%(obs_info['Mean_SEFD'],obs_info['psrflux_Sens'])
            self.write(info_str)
            self.write('--------------------------\n')


    def report_tophit(self, max_val, ind, ind_tuple, tdwidth, fftlen, header,spec_slice=None,obs_info=None):

        '''This function looks into the top hit in a region, basically find the local maximum and saves that.
        '''
        if not self.tophit_count:
            self.report_header(header,obs_info=obs_info)
            info_str = 'N_candidates: %i \n'%max_val.total_n_candi
            self.write(info_str)
            info_str = '--------------------------\n'
            info_str += 'Top Hit # \t'
            info_str += 'Drift Rate \t'
            info_str += 'SNR \t'
            info_str += 'Uncorrected Frequency \t'
            info_str += 'Corrected Frequency \t'
            info_str += 'Index \t'
            info_str += 'freq_start \t'
            info_str += 'freq_end \t'
            info_str += 'SEFD \t'
            info_str += 'SEFD_freq \t'
            info_str +='\n'
            self.write(info_str)
            self.write('--------------------------\n')

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
        info_str +='\n'
        self.write(info_str)


#EE I dont like this nested for loops, makes it slower, also, really need this formating of the array?
#EE But may need it ( I'm doing the same in drift_index_test.)
#EE Well, then maybe need to check if need to save all the time...
#EE Or maybe the format makes it be too large. Should save as binary instead...or fits?

#         for i in range(0, spec_slice.shape[0]):
#             info_str = ''
#             for j in range(0, spec_slice.shape[-1]):
#                 info_str += '%14.6f '%spec_slice[i, j]
#             info_str += '\n'
#             self.write(info_str)
#         self.write('\n')

        return self


class LogWriter(GeneralWriter):
    """ """
    def report_candidate(self, info_str):
        return None

    def info(self, info_str):
        self.write(info_str + '\n')
        return None



