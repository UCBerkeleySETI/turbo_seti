#!/usr/bin/env python

import numpy as np
from .helper_functions import chan_freq


class GeneralWriter:
    r"""
    Wrapper class for file operations.
    
    """
    def __init__(self, filename='', mode='a'):
        r"""
        Initializes GeneralWriter object. Opens given file with given mode, sets new object's filehandle to the file
        object, sets the new object's filename to the file's name, then closes the file.

        Parameters
        ----------
        filename : str
            Name of file on which we would like to perform operations.
        mode : str {'a', 'r', 'w', 'x'}, optional
            Mode which we want to use to open file, same modes as the built-in python
            built-in open function: read (`r`), append (`a`), write (`w`), or create (`x`).

        """
        with open(filename, mode) as myfile:
            self.filehandle = myfile
            self.filename = filename

    def close(self):
        r"""
        Closes file object if it is open.

        """
        if self.filehandle.closed:
            pass
        else:
            self.filehandle.close()

    def open(self, mode='a'):
        r"""
        Opens the file with the inputted mode, then closes it. Does not actually leave the file opened, only used for
        changing mode.

        Parameters
        ----------
        mode : str {'a', 'r', 'w', 'x'}, optional
            Mode which we want to assign to this file, same modes as the built-in python
            built-in open function: read (`r`), append (`a`), write (`w`), or create (`x`).

        """
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
        r"""
        Checks if file is open.

        Returns
        ------- 
          : boolean
            True if file is open, False otherwise.

        """
        return not self.filehandle.closed

    def writable(self):
        r"""
        Checks if file is open, and if it is, checks that mode is either write or append.

        Returns
        ------- 
          : boolean
            True if file is open and writeable, False otherwise.

        """
        return self.is_open() and (('w' in self.filehandle.mode) or ('a' in self.filehandle.mode))

    def write(self, info_str, mode='a'):
        r"""
        Sets file mode to a writeable mode and opens it if it is not already open in a writeable mode, writes info_str
        to it, and then closes it. If the file was not previously open when this is called, the file is closed after
        writing in order to maintain the state the filewriter was in before.

        Parameters
        ----------
        info_str : str
            Data to be written to file.
        mode : str {'a', 'w'}, optional
            Mode for file. If it is not a writeable mode, it will be set to a writeable mode.

        """
        if mode not in ('a', 'w'):
            mode = 'a'
        if not self.writable():
            with open(self.filename, mode) as myfile:
                myfile.write(info_str)
                self.filehandle = myfile
        else:
            self.filehandle.write(info_str)


class FileWriter(GeneralWriter):
    r"""
    Used to write information to turboSETI output files.
    
    """
    def __init__(self, filename, header):
        r"""
        Initializes FileWriter object and writes its header.

        Parameters
        ----------
        filename : str
            Name of file on which we would like to perform operations.
        header : dict
            Information to be written to header of file filename.

        """
        GeneralWriter.__init__(self, filename)
        self.write('# -------------------------- o --------------------------\n')
        self.write('# File ID: %s \n'%(filename.split('/')[-1].replace('.dat','')+'.h5'))
        self.write('# -------------------------- o --------------------------\n')
        self.report_header(header)

        self.tophit_count = 0

    def report_header(self, header):
        r"""
        Write header information per given obs.

        Parameters
        ----------
        header : dict
            Information to be written to file header.

        """
        info_str = '# Source:{}\n# MJD: {:18.12f}\tRA: {}\tDEC: {}\n# DELTAT: {:10.6f}\tDELTAF(Hz): {:10.6f}\tmax_drift_rate: {:10.6f}\tobs_length: {:10.6f}\n' \
                   .format(header['SOURCE'],header['MJD'], header['RA'], header['DEC'], header['DELTAT'], header['DELTAF']*1e6, header['max_drift_rate'], header['obs_length'])

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

    def report_tophit(self, max_val, ind, ind_tuple, tdwidth, fftlen, header, total_n_candi, obs_info=None):
        r"""
        This function looks into the top hit in a region, basically finds the local maximum and saves that.

        Parameters
        ----------
          max_val : findopp
          ind : int
            Index at which top hit is located in max_val's maxdrift and maxsnr.
          ind_tuple: tuple(int, int) (lbound, ubound)
          tdwidth : int
          fftlen : int
            Length of the fast fourier transform matrix.
          header : dict
            Contains info on coarse channel to be written to file.
          total_n_candi : int
          obs_info:  dict, optional
            Used to hold info found on file, including info about pulsars, RFI, and SEFD.

        Returns
        -------
          : FileWriter object that called this function.

        """
        offset = int((tdwidth - fftlen)/2)
        tdwidth =  len(max_val.maxsnr)

        self.tophit_count += 1
        freq_start = chan_freq(header, ind_tuple[0]-offset, tdwidth, 0)
        freq_end   = chan_freq(header, ind_tuple[1]-1-offset, tdwidth, 0)

        uncorr_freq = chan_freq(header, ind-offset, tdwidth, 0)
        corr_freq = chan_freq(header, ind-offset, tdwidth, 1)

        #Choosing the index of given SEFD and freq.
        if obs_info['SEFDs_freq'][0] > 0.:
            this_one = np.arange(len(obs_info['SEFDs_freq']))[ (obs_info['SEFDs_freq_up']>uncorr_freq) ][0]
        else:
            this_one = 0

        info_str = '%06d\t'%(self.tophit_count)  #Top Hit number
        info_str += '%10.6f\t'%max_val.maxdrift[ind]  #Drift Rate
        info_str += '%10.6f\t'%max_val.maxsnr[ind]  #SNR
        info_str += '%14.6f\t'%uncorr_freq #Uncorrected Frequency:
        info_str += '%14.6f\t'%corr_freq #Corrected Frequency:
        info_str += '%d\t'%(ind - offset) #Index:
        info_str += '%14.6f\t'%freq_start #freq_start:
        info_str += '%14.6f\t'%freq_end #freq_end:
        info_str += '%s\t'%obs_info['SEFDs_val'][this_one] #SEFD:
        info_str += '%14.6f\t'%obs_info['SEFDs_freq'][this_one] #SEFD_mid_freq:
        info_str += '%i\t'%header['cchan_id']
        info_str += '%i\t'%total_n_candi #
        info_str +='\n'
        self.write(info_str)

        return self

class LogWriter(GeneralWriter):
    r"""
    Used to write data to log.

    """
    def info(self, info_str):
        r"""
        Writes info_str to file.

        Parameters
        ----------
        info_str : str
            String to be written to file.

        """
        self.write(info_str + '\n')
