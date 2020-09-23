#!/usr/bin/env python

import numpy as np
from .helper_functions import chan_freq

from logbook import Logger
from ..log import logger_group
logger = Logger('turboseti.file_writers')
logger_group.add_logger(logger)

class GeneralWriter:
    """Wrapper class for file operations."""
    def __init__(self, filename='', mode='a'):
        """
        Initializes GeneralWriter object. Opens given file with given mode, sets new object's filehandle to the file
        object, sets the new object's filename to the file's name, then closes the file.

        Args:
            filename:    string,     name of file on which we would like to perform operations
            mode:        string,     mode which we want to use to open file, same modes as the built-in python
                                      built-in open function: r - read, a - append, w -write, x - create
        """
        with open(filename, mode) as myfile:
            self.filehandle = myfile
            self.filename = filename

    def close(self):
        """Closes file object if it is open.
        """
        if self.filehandle.closed:
            pass
        else:
            self.filehandle.close()

    def open(self, mode='a'):
        """Opens the file with the inputted mode, then closes it. Does not actually leave the file opened, only used for
        changing mode.

        Args:
          mode: string,     mode which we want to assign to this file, same modes as the built-in python
        built-in open function: r - read, a - append, w -write, x - create (Default value = 'a')
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
        """Checks if file is open.
        Returns:  boolean,    true if file is open, false otherwise
        """
        return not self.filehandle.closed

    def writable(self):
        """Checks if file is open, and if it is, checks that mode is either write or append.
        :return:    boolean,    true if file is open and writeable, false otherwise

        Args:

        Returns:

        """
        return self.is_open() and (('w' in self.filehandle.mode) or ('a' in self.filehandle.mode))

    def write(self, info_str, mode='a'):
        """Sets file mode to a writeable mode and opens it if it is not already open in a writeable mode, writes info_str
        to it, and then closes it. If the file was not previously open when this is called, the file is closed after
        writing in order to maintain the state the filewriter was in before.

        Args:
          info_str: string,     data to be written to file
          mode: string,     mode for file. If it is not a writeable mode, it will be set to a writeable mode (Default value = 'a')

        Returns:

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
    """Used to write information to turboSETI output files."""
    def __init__(self, filename, header):
        """ Initializes FileWriter object and writes its header.

        :param filename:    string,     name of file on which we would like to perform operations
        :param header:      dict,       information to be written to header of file filename
        """
        GeneralWriter.__init__(self, filename)
        self.write('# -------------------------- o --------------------------\n')
        self.write('# File ID: %s \n'%(filename.split('/')[-1].replace('.dat','')+'.h5'))
        self.write('# -------------------------- o --------------------------\n')
        self.report_header(header)

        self.tophit_count = 0

    def report_header(self, header):
        """Write header information per given obs.

        Args:
          header: dict,       information to be written to file header

        Returns:
          : void

        """

        info_str = '# Source:%s\n# MJD: %18.12f\tRA: %s\tDEC: %s\n# DELTAT: %10.6f\tDELTAF(Hz): %10.6f\n'%\
                   (header['SOURCE'],header['MJD'], header['RA'], header['DEC'], header['DELTAT'], header['DELTAF']*1e6)

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
        info_str += 'Fscrunch_level \t'
        info_str +='\n'
        self.write(info_str)
        self.write('# --------------------------\n')

    def report_tophit(self, max_val, ind, ind_tuple, tdwidth, fftlen, header, total_n_candi, fscrunch, obs_info=None):
        """This function looks into the top hit in a region, basically finds the local maximum and saves that.

        Args:
          max_val: findopp.max_vals,
          ind: int,                    index at which top hit is located in max_val's maxdrift and
        maxsnr
          ind_tuple: tuple(int, int)         (lbound, ubound)
          tdwidth: int,
          fftlen: int,                    length of the fast fourier transform matrix
          header: dict,                   contains info on coarse channel to be written to file
          total_n_candi: int,
          ### spec_slice: dict, (Default value = None) <--- UNUSED
          obs_info: dict,                   used to hold info found on file, including info about pulsars,
        RFI, and SEFD (Default value = None)

        Returns:
          : FileWriter object that called this function.

        """
        offset = int((tdwidth - fftlen)/2)

        ind_fullres = ind * fscrunch
        offset_fullres = offset
        tdwidth_fullres = tdwidth
        lb, ub = ind_tuple
        lb_fullres, ub_fullres = lb * fscrunch, ub * fscrunch

        self.tophit_count += 1

        freq_start = chan_freq(header, lb_fullres-offset_fullres, tdwidth_fullres, 0)
        freq_end   = chan_freq(header, ub_fullres-offset_fullres, tdwidth_fullres, 0)
        logger.debug(f"{lb_fullres}, {ub_fullres}, {tdwidth_fullres}, {freq_start}, {freq_end}")
        uncorr_freq = chan_freq(header, ind_fullres-offset_fullres, tdwidth_fullres, 0)
        corr_freq = chan_freq(header, ind_fullres-offset_fullres, tdwidth_fullres, 1)

        #Choosing the index of given SEFD and freq.
        if obs_info['SEFDs_freq'][0] > 0.:
            this_one = np.arange(len(obs_info['SEFDs_freq']))[ (obs_info['SEFDs_freq_up']>uncorr_freq) ][0]
        else:
            this_one = 0

        info_str = '%03d\t'%(self.tophit_count)  #Top Hit number
        info_str += '%10.6f\t'%max_val.maxdrift[ind]  #Drift Rate
        info_str += '%10.6f\t'%max_val.maxsnr[ind]  #SNR
        info_str += '%14.10f\t'%uncorr_freq #Uncorrected Frequency:
        info_str += '%14.10f\t'%corr_freq #Corrected Frequency:
        info_str += '%d\t'%(ind_fullres - offset_fullres) #Index:
        info_str += '%14.10f\t'%freq_start #freq_start:
        info_str += '%14.10f\t'%freq_end #freq_end:
        info_str += '%s\t'%obs_info['SEFDs_val'][this_one] #SEFD:
        info_str += '%14.6f\t'%obs_info['SEFDs_freq'][this_one] #SEFD_mid_freq:
        info_str += '%i\t'%header['coarse_chan']
        info_str += '%i\t'%total_n_candi #
        info_str += '%i\t' % fscrunch  #
        info_str +='\n'
        self.write(info_str)

        return self

class LogWriter(GeneralWriter):
    """Used to write data to log."""
    def info(self, info_str):
        """Writes info_str to file.

        Args:
          info_str: string,     to be written to file

        Returns:
          : void

        """
        self.write(info_str + '\n')
