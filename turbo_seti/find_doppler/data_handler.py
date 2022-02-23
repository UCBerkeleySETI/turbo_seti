"""
Filterbank data handler for the find_doppler.py functions.
"""

import os
import math
import logging

import numpy as np
from pkg_resources import resource_filename
import h5py

from blimpy import Waterfall
from blimpy.io import sigproc
from blimpy.io.hdf_writer import __write_to_hdf5_heavy as write_to_h5
from blimpy.utils import change_the_ext

from .kernels import Kernels

logger = logging.getLogger('data_handler')

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
        Use GPU accelerated Kernels?
    precision : int {2: float64, 1: float32}, optional
        Floating point precision.  Default: 1.
    gpu_id : int
        If  gpu_backend=True, then this is the device ID to use.
    """
    def __init__(self, filename=None, out_dir='./', n_coarse_chan=None, coarse_chans=None,
                 kernels=None, gpu_backend=False, precision=1, gpu_id=0):
        if not kernels:
            self.kernels = Kernels(gpu_backend, precision, gpu_id)
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
                    raise IOError(errmsg)
                logger.info("Filterbank file detected. Attempting to create .h5 file in current directory...")
                self.__make_h5_file()

            self.filestat = os.stat(filename)
            self.filesize = self.filestat.st_size/(1024.0**2)

            # Grab header from DATAH5
            datah5_obj = DATAH5(filename, kernels=self.kernels, gpu_id=gpu_id)
            self.header = datah5_obj.header
            self.drift_rate_resolution = datah5_obj.drift_rate_resolution
            datah5_obj.close()

            # Split the file
            self.cchan_list = self.__split_h5()
            self.status = True

        else:
            self.status = False
            errmsg = "File {} doesn\'t exist, please check!".format(filename)
            raise IOError(errmsg)

    def get_info(self):
        r"""
        Get the header of the file.

        Returns
        -------
        header : dict
            Header of the blimpy file.

        """
        wf = Waterfall(self.filename, load_data=False)
        return wf.header

    def __make_h5_file(self):
        r"""
        Converts file to h5 format, saved in current directory.
        Sets the filename attribute of the calling DATAHandle.
        to the (new) filename.

        """
        wf = Waterfall(self.filename, load_data=False)
        fil_path = os.path.basename(self.filename)
        h5_path = os.path.join(self.out_dir, change_the_ext(fil_path, 'fil', 'h5'))
        try:
            os.remove(h5_path)
        except:
            pass
        write_to_h5(wf, h5_path)
        self.filename = h5_path

    def __split_h5(self):
        r"""
        Creates a plan to select data from single coarse channels.

        Returns
        -------
        chan_list : list
            Where each list member contains a coarse channel dict object
            for each coarse channel in the file.

            Dict fields:
            * filename : file path (common to all objects)
            * f_start : start frequency of coarse channel
            * f_stop : stop frequency of coarse channel
            * coarse_chan_id : coarse channel number (identifier)
            * n_coarse_chan : total number of coarse channels (common to all objects)

        """
        cchan_list = []

        # Create a Waterfall object
        try:
            wf = Waterfall(self.filename, load_data=False)
        except:
            errmsg = "Error encountered when trying to open file: {}".format(self.filename)
            raise IOError(errmsg)

        #Finding lowest freq in file.
        f_delt = wf.header['foff']
        f0 = wf.header['fch1']

        # Determine the number of coarse channels: user, blimpy, of header nfpc field.
        if self.n_coarse_chan is not None: # specifictaion from user
            logger.info("From user, n_coarse_chan={}".format(self.n_coarse_chan))
        else: # no specifictaion from user
            self.nfpc = wf.header.get('nfpc', None) # Store nfpc value in the DATAHandle object.
            if self.nfpc is not None: # nfpc is specified in file header
                if self.nfpc < 1: # nfpc valid?
                    errmsg = "Filterbank header field NFPC must be > 0 but I saw {}.  Defaulting to blimpy!" \
                             .format(self.nfpc)
                    logger.warning(errmsg)
                    self.n_coarse_chan = int(wf.calc_n_coarse_chan())
                    logger.info("From blimpy, n_coarse_chan={}"
                                .format(self.n_coarse_chan))
                else:  # nfpc is valid.  Use it to calculate the number of coarse channels.
                    self.n_coarse_chan = int(0.01 + wf.header['nchans'] / self.nfpc)
                    logger.info("From nfpc={}, n_fine_nchans={} and n_coarse_chan={}"
                                .format(self.nfpc, wf.header['nchans'], self.n_coarse_chan))
            else: # nfpc is NOT in the file header, not specified by user.
                self.n_coarse_chan = int(wf.calc_n_coarse_chan())
                logger.info("From blimpy, n_coarse_chan={}"
                            .format(self.n_coarse_chan))

        # Only load coarse chans of interest -- or do all if not specified
        if self.coarse_chans in (None, ''):
            self.coarse_chans = range(self.n_coarse_chan)

        for cchan_id in self.coarse_chans:

            # Calculate the frequency range for the given course channel (cchan_id).
            f_start = f0 + cchan_id * (f_delt) * wf.n_channels_in_file / self.n_coarse_chan
            f_stop = f0 + (cchan_id + 1) * (f_delt) * wf.n_channels_in_file / self.n_coarse_chan

            if f_start > f_stop:
                f_start, f_stop = f_stop, f_start

            # Instantiate the coarse channel object.
            cchan_obj = {'filename': self.filename,
                        'f_start': f_start,
                        'f_stop': f_stop,
                        'cchan_id': cchan_id,
                        'n_coarse_chan': self.n_coarse_chan}

            # Append coarse channel object to list.
            cchan_list.append(cchan_obj)

        return cchan_list

class DATAH5:
    r"""
    This class is where the waterfall data is loaded, as well as the DATAH5 header info.
    Don't be surprised at the use of FITS header names! [?]
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
        Coarse channel ID.
    n_coarse_chan : int
        Total number of coarse channels.
    kernels : Kernels
        Pre-configured class of kernels.

    """
    def __init__(self, filename, f_start=None, f_stop=None, t_start=None, t_stop=None,
                 cchan_id=0, n_coarse_chan=None, kernels=None, gpu_backend=False, precision=1, gpu_id=0):
        self.filename = filename
        self.closed = False
        self.f_start = f_start
        self.f_stop = f_stop
        self.t_start = t_start
        self.t_stop = t_stop
        self.n_coarse_chan = n_coarse_chan

        if not kernels:
            self.kernels = Kernels(gpu_backend, precision, gpu_id)
        else:
            self.kernels = kernels

        # Create a Waterfall object instance.
        try:
            self.fil_file = Waterfall(filename, f_start=self.f_start, f_stop=self.f_stop,
                                      t_start=self.t_start, t_stop=self.t_stop, load_data=False)
        except:
            errmsg = "Error encountered when trying to open file: {}".format(filename)
            raise IOError(errmsg)

        # Create a header used by the search.
        try:
            if self.n_coarse_chan:
                header = self.__make_data_header(self.fil_file.header, coarse=True)
            else:
                header = self.__make_data_header(self.fil_file.header)
        except:
            errmsg = "Error accessing header from file: {}".format(self.fil_file.header)
            raise IOError(errmsg)

        self.header = header

        self.fftlen = header['NAXIS1']

        #EE To check if swapping tsteps_valid and tsteps is more appropriate.
        self.tsteps_valid = int(self.fil_file.n_ints_in_file)
        self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))

        self.header['obs_length'] = self.tsteps_valid * header['DELTAT']
        self.drift_rate_resolution = (1e6 * np.abs(header['DELTAF'])) / self.header['obs_length']   # in Hz/sec
        self.header['cchan_id'] = cchan_id

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

        dim_time = self.fil_file.data.shape[0]
        if dim_time < 2:
            msg = "data_handler.py:load_data: Cannot handle data with only 1 time step!"
            logger.error(msg)
            msg = "data shape = {}!".format(self.fil_file.data.shape)
            raise ValueError(msg)

        spectra = self.kernels.np.squeeze(self.fil_file.data)

        # This check will add rows of zeros if the obs is too short
        # (and thus not a power of two rows).
        if spectra.shape[0] < self.tsteps:
            padding = self.kernels.np.zeros((self.tsteps-spectra.shape[0], self.fftlen))
            spectra = self.kernels.np.concatenate((spectra, padding), axis=0)

        self.tsteps_valid = self.tsteps
        #updating obs_length and drift_rate resolution after changing the spectra shape
        self.header['obs_length'] = self.tsteps * self.header['DELTAT']
        self.drift_rate_resolution = (1e6 * np.abs(self.header['DELTAF'])) / self.header['obs_length']   # in Hz/sec

        if spectra.shape != (self.tsteps_valid, self.fftlen):
            msg = "data_handler.py:load_data: spectra.shape={}!".format(spectra.shape)
            logger.error(msg)
            msg = "data_handler.py:load_data: tsteps_valid={}, fftlen={}!" \
                    .format(self.tsteps_valid, self.fftlen)
            raise ValueError(msg)

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
        dia_num = int(np.log2(self.tsteps))
        file_path = resource_filename('turbo_seti', f'drift_indexes/drift_indexes_array_{dia_num}.txt')

        if not os.path.isfile(file_path):
            dia_file = 'drift_indexes/drift_indexes_array_{}.txt'.format(dia_num)
            msg = "data_handler.py:load_drift_indexes: time integration steps = {}!".format(self.tsteps)
            logger.error(msg)
            msg = "data_handler.py:load_drift_indexes: file {} not found!".format(dia_file)
            logger.error(msg)
            if self.tsteps < 4:
                raise ValueError("Number of time integration steps must be at least 4!")
            raise ValueError("""Don't attempt to use High Time Resolution (HTR) files with turboSETI. """
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
            base_header['FCNTR'] = self.kernels.np.abs(self.f_stop - self.f_start) / 2. + self.kernels.np.fmin(
                self.f_start, self.f_stop)
        else:
            base_header['NAXIS1'] = int(header['nchans'])
            base_header['FCNTR'] = float(header['fch1']) + header['foff'] * base_header['NAXIS1'] / 2

        # Return base_header to caller.
        return base_header

    def close(self):
        r"""
        Closes file and sets the data attribute `.closed` to
        True. A closed object can no longer be used for I/O operations.
        `close()` may be called multiple times without error.

        """
        # Call file object destructor which should close the file
        if hasattr(self, 'fil_file'):
            del self.fil_file

        if hasattr(self, 'kernels'):
            del self.kernels

        self.closed = True
