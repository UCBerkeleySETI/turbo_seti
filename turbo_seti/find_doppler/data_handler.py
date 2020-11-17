#!/usr/bin/env python

import os
import math
import logging

import numpy as np
from pkg_resources import resource_filename
import h5py

from blimpy import Waterfall
from blimpy.io import sigproc

from .kernels import Kernels

logger = logging.getLogger(__name__)

#For debugging
#import cProfile
#import pdb;# pdb.set_trace()

class DATAHandle:
    r"""
    Class to setup input file for further processing of data.
    Handles conversion to h5 (from fil), extraction of
    coarse channel info, waterfall info, and file size checking.

    Parameters
    ----------
    filename : str
        Name of file (.h5 or .fil).
    out_dir : str
        Directory where output files should be saved.
    n_coarse_chan : int
        Number of coarse channels.
    coarse_chans : list or None
        List of course channels.
    kernels : Kernels, optional
        Pre-configured class of Kernels.
    gpu_backend : bool, optional
        Use GPU accelerated Kernels.
    precision : int {2: float64, 1: float32}, optional
        Floating point precision.

    """
    def __init__(self, filename=None, out_dir='./', n_coarse_chan=None, coarse_chans=None,
                 kernels=None, gpu_backend=False, precision=2):
        if not kernels:
            self.kernels = Kernels(gpu_backend, precision)
        else:
            self.kernels = kernels

        if filename and os.path.isfile(filename):
            self.filename = filename
            self.out_dir = out_dir
            self.n_coarse_chan = n_coarse_chan
            self.coarse_chans = coarse_chans

            if not h5py.is_hdf5(filename):
                if not sigproc.is_filterbank(filename):
                    self.status = False
                    errmsg = 'Not a filterbank file: {}'.format(filename)
                    logger.error(errmsg)
                    raise IOError(errmsg)
                logger.info("Filterbank file detected. Attempting to create .h5 file in current directory...")
                try:
                    self.__make_h5_file()
                except:
                    self.status = False
                    errmsg = 'Unable to create .h5 file from: {}'.format(filename)
                    logger.error(errmsg)
                    raise IOError(errmsg)

            self.filestat = os.stat(filename)
            self.filesize = self.filestat.st_size/(1024.0**2)

            # Grab header from DATAH5
            dobj_master = DATAH5(filename, n_coarse_chan=self.n_coarse_chan, kernels=self.kernels)
            self.header = dobj_master.header
            dobj_master.close()

            # Split the file
            self.data_list = self.__split_h5()
            self.status = True

        else:
            self.status = False
            errmsg = "File {} doesn\'t exist, please check!".format(filename)
            logger.error(errmsg)
            raise IOError(errmsg)

    def get_info(self):
        r"""
        Get the header of the file.

        Returns
        -------
        header : dict
            Header of the blimpy file.

        """
        fil_file = Waterfall(self.filename, load_data=False)
        return fil_file.header

    def __make_h5_file(self):
        r"""
        Converts file to h5 format, saved in current directory.
        Sets the filename attribute of the calling DATAHandle.
        to the (new) filename.

        """
        fil_file = Waterfall(self.filename)
        bn = os.path.basename(self.filename)
        new_filename = os.path.join(self.out_dir, bn.replace('.fil', '.h5'))
        fil_file.write_to_hdf5(new_filename)
        self.filename = new_filename

    def __split_h5(self):
        r"""
        Creates a plan to select data from single coarse channels.

        Returns
        -------
        data_list : list[DATAH5]
            Where each list member contains a DATAH5 object
            for each of the coarse channels in the file.

        """
        data_list = []

        #Instancing file.
        try:
            fil_file = Waterfall(self.filename)
        except:
            errmsg = "Error encountered when trying to open file: {}".format(self.filename)
            logger.error(errmsg)
            raise IOError(errmsg)

        #Finding lowest freq in file.
        f_delt = fil_file.header['foff']
        f0 = fil_file.header['fch1']

        #Looping over the number of coarse channels.
        if self.n_coarse_chan is not None:
            n_coarse_chan = self.n_coarse_chan
        elif fil_file.header.get('n_coarse_chan', None) is not None:
            n_coarse_chan = fil_file.header['n_coarse_chan']
        else:
            n_coarse_chan = int(fil_file.calc_n_coarse_chan())

        # Only load coarse chans of interest -- or do all if not specified
        if self.coarse_chans in (None, ''):
            self.coarse_chans = range(n_coarse_chan)

        for chan in self.coarse_chans:

            #Calculate freq range for given course channel.
            f_start = f0 + chan * (f_delt) * fil_file.n_channels_in_file / n_coarse_chan
            f_stop = f0 + (chan + 1) * (f_delt) * fil_file.n_channels_in_file / n_coarse_chan

            if f_start > f_stop:
                f_start, f_stop = f_stop, f_start

            data_obj = {'filename': self.filename,
                        'f_start': f_start,
                        'f_stop': f_stop,
                        'coarse_chan': chan,
                        'n_coarse_chan': n_coarse_chan}

            #This appends to a list of all data instance selections. So that all get processed later.
            data_list.append(data_obj)

        return data_list

class DATAH5:
    r"""
    This class is where the waterfall data is loaded, as well as the header info.
    It creates other attributes related to the dedoppler search (load_drift_indexes).

    Parameters
    ----------
    filename : string
        Name of file.
    f_start : float
        Start frequency in MHz.
    f_stop : float
        Stop frequency in MHz.
    t_start : int
        Start integration ID.
    t_stop : int
        Stop integration ID.
    coarse_chan : int
    n_coarse_chan : int
    kernels : Kernels
        Pre-configured class of kernels.

    """
    def __init__(self, filename, f_start=None, f_stop=None, t_start=None, t_stop=None,
                 coarse_chan=1, n_coarse_chan=None, kernels=None):
        self.filename = filename
        self.closed = False
        self.f_start = f_start
        self.f_stop = f_stop
        self.t_start = t_start
        self.t_stop = t_stop
        self.n_coarse_chan = n_coarse_chan
        self.kernels = kernels

        #Instancing file.
        try:
            self.fil_file = Waterfall(filename, f_start=self.f_start, f_stop=self.f_stop,
                                      t_start=self.t_start, t_stop=self.t_stop, load_data=False)
        except:
            errmsg = "Error encountered when trying to open file: {}".format(filename)
            logger.error(errmsg)
            raise IOError(errmsg)

        #Getting header
        try:
            if self.n_coarse_chan:
                header = self.__make_data_header(self.fil_file.header, coarse=True)
            else:
                header = self.__make_data_header(self.fil_file.header)
        except:
            errmsg = "Error accessing header from file: {}".format(self.fil_file.header)
            logger.error(errmsg)
            raise IOError(errmsg)

        self.header = header

        self.fftlen = header['NAXIS1']

        #EE To check if swapping tsteps_valid and tsteps is more appropriate.
        self.tsteps_valid = header['NAXIS2']
        self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))

        self.obs_length = self.tsteps_valid * header['DELTAT']
        self.drift_rate_resolution = (1e6 * np.abs(header['DELTAF'])) / self.obs_length   # in Hz/sec
        self.header['baryv'] = 0.0
        self.header['barya'] = 0.0
        self.header['coarse_chan'] = coarse_chan

        #EE For now I'm not using a shoulder. This is ok as long as
        ##  I'm analyzing each coarse channel individually.
        #EE In general this parameter is an integer (even number).
        #This gives two regions, each of n*steps, around spectra[i]
        self.shoulder_size = 0
        self.tdwidth = self.fftlen + self.shoulder_size * self.tsteps

    def load_data(self):
        r"""
        Read the spectra and drift indices from file.

        Returns
        -------
        spectra, drift indices : ndarray, ndarray

        """
        self.fil_file.read_data(f_start=self.f_start, f_stop=self.f_stop)

        #Blanking DC bin.
        if self.n_coarse_chan is not None:
            n_coarse_chan = self.n_coarse_chan
        else:
            n_coarse_chan = int(self.fil_file.calc_n_coarse_chan())
        self.fil_file.blank_dc(n_coarse_chan)

        spec = self.kernels.xp.squeeze(self.fil_file.data)
        spectra = self.kernels.xp.array(spec, dtype=self.kernels.float_type)

        # DCP APR 2020 -- COMMENTED OUT. THIS IS BREAKING STUFF IN CURRENT VERSION.
        #Arrange data in ascending order in freq if not already in that format.
        #if self.header['DELTAF'] < 0.0:
        #    spectra = spectra[:,::-1]

        # This check will add rows of zeros if the obs is too short
        # (and thus not a power of two rows).
        if spectra.shape[0] != self.tsteps:
            padding = self.kernels.xp.zeros((self.tsteps-spectra.shape[0], self.fftlen))
            spectra = self.kernels.xp.concatenate((spectra, padding), axis=0)

        self.tsteps_valid = self.tsteps
        self.obs_length = self.tsteps * self.header['DELTAT']

        if spectra.shape != (self.tsteps_valid, self.fftlen):
            logger.error('Something is wrong with array size.')
            raise IOError('Something is wrong with array size.')

        drift_indexes = self.load_drift_indexes()

        return spectra, drift_indexes

    def load_drift_indexes(self):
        r"""
        The drift indices are read from a stored file so that
        there is no need to recalculate. This speed things up.

        Returns
        -------
        drift_indexes : ndarray

        """
        n = int(np.log2(self.tsteps))
        file_path = resource_filename('turbo_seti', f'drift_indexes/drift_indexes_array_{n}.txt')

        if not os.path.isfile(file_path):
            raise ValueError("""Don't attempt to use High Time Resolution (HRT) files with turboSETI. """
                             """TurboSETI is designed to search for narrowband signals -- the maximum """
                             """doppler drift we can expect due to the motion of celestial bodies is a few Hz/s. """
                             """The high time resolution products (ending 0001.fil) has ~0.5 MHz resolution and """
                             """~100 us integrations, so you'd be looking at insane drift rates. Issue #117.""")

        di_array = np.array(np.genfromtxt(file_path, delimiter=' ', dtype=int))

        ts2 = int(self.tsteps/2)
        drift_indexes = di_array[(self.tsteps_valid - 1 - ts2), 0:self.tsteps_valid]
        return drift_indexes

    def __make_data_header(self, header, coarse=False):
        r"""
        Takes header into fits header format.

        Parameters
        ----------
        header : dict
            Blimpy waterfall header.
        coarse : Boolean
            Whether or not there are coarse channels to analyze.

        Returns
        -------
        base_header : dict

        """
        base_header = {}

        #used by file_writers.py
        base_header['SOURCE'] = header['source_name']
        base_header['MJD'] = header['tstart']
        base_header['DEC'] = str(header['src_dej'])
        base_header['RA'] = str(header['src_raj'])
        base_header['DELTAF'] = header['foff']
        base_header['DELTAT'] = float(header['tsamp'])

        #used by helper_functions.py
        if coarse:
            base_header['NAXIS1'] = int(header['nchans']/self.n_coarse_chan)
        else:
            base_header['NAXIS1'] = int(header['nchans'])
        base_header['FCNTR'] = float(header['fch1']) + header['foff'] * base_header['NAXIS1'] / 2

        #other header values.
        base_header['NAXIS'] = 2
        base_header['NAXIS2'] = int(self.fil_file.n_ints_in_file)
        return base_header

    def close(self):
        r"""
        Closes file and sets the data attribute `.closed` to
        True. A closed object can no longer be used for I/O operations.
        `close()` may be called multiple times without error.

        """
        # Call file object destructor which should close the file
        del self.fil_file

        self.closed = True
