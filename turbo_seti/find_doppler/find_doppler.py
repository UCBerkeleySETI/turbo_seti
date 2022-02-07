"""
turbo_seti doppler search module
================================

This module is deeply dependent on classes and functions in data_handler.py.

Main class: FindDoppler

Independent functions:
    search_coarse_channel - for a given coarse channel, doppler search.
    load_the_data - loads everything needed by search_coarse_channel.
    populate_tree - populate "tree_findoppler" used by several functions.
    hitsearch - Searches for hits at given drift rate.
    tophitsearch - Searches for hits with largest SNR within 2*tsteps fine frequency channels.
"""

import os
import time
import logging

import dask.bag as db
from dask.diagnostics import ProgressBar

from h5py.version import hdf5_version as LIBHDF5_VERSION
from h5py import __version__ as H5PY_VERSION
from h5py._errors import unsilence_errors
from hdf5plugin import version as HDF5PLUGIN_VERSION
from blimpy import __version__ as BLIMPY_VERSION
from .turbo_seti_version import TURBO_SETI_VERSION
from .kernels import Kernels, Scheduler
from .data_handler import DATAHandle, DATAH5
from .file_writers import FileWriter, LogWriter
from .helper_functions import chan_freq, comp_stats
from .merge_dats_logs import merge_dats_logs

#For debugging
#import pdb;# pdb.set_trace()
logger_name = 'find_doppler'
logger = logging.getLogger(logger_name)
version_announcements = '\nturbo_seti version {}\nblimpy version {}\nh5py version {}\nhdf5plugin version {}\nHDF5 library version {}\n\n' \
                        .format(TURBO_SETI_VERSION, BLIMPY_VERSION, H5PY_VERSION, HDF5PLUGIN_VERSION, LIBHDF5_VERSION)

class max_vals:
    r"""
    Class used to initialize some maximums.
    """
    def __init__(self):
        # For each index value of the FFT length:
        self.maxsnr = None        # Signal with the maximum SNR
        self.maxdrift = None      # Drift rate of signal
        # This is an array of ONE element:
        self.total_n_hits = None  # Accumulated total number of hits.


class FindDoppler:
    r"""
    Initializes FindDoppler object.

    Parameters
    ----------
    datafile : string
        Input filename (.h5 or .fil)
    max_drift : float
        Maximum drift rate in Hz/second.
    min_drift : float
        Minimum drift rate in Hz/second.
    snr : float
        Minimum Signal to Noise Ratio (SNR) - A ratio bigger than 1 to 1 has more signal than noise.
    out_dir : string
        Directory where output files should be placed. By default this is the
        current working directory.
    coarse_chans : list(int)
        The input comma-separated list of coarse channels to analyze, if any. By default,
        all coarse channels will be searched. Use this to search only specified channels,
        e.g. [7,12] will search channels 7 and 12 only.
    obs_info : dict
        Used to hold information found on file, including info about pulsars, RFI, and SEFD.
    flagging : bool
        Flags the edges of the PFF for BL data (with 3Hz res per channel)? (True/False)
        Anybody - please improve this cryptic description.
    n_coarse_chan : int
        Number of coarse channels in the file.
        If None (default), blimpy will make this determination (undesirable, in general).
    kernels : Kernels, optional
        Pre-configured class of Kernels.
    gpu_backend : bool, optional
        Use GPU accelerated Kernels? (True/False)
    gpu_id : int
        If gpu_backend=True, then this is the GPU device to use.
        Default is 0.
    precision : int {2: float64, 1: float32}, optional
        Floating point precision for the GPU.
        The default is 1 (recommended).
    append_output : bool, optional
        Append output DAT & LOG files? (True/False)
        Default is False.
        DEPRECATED.
    log_level_int : int, optional
        Python logging threshold level (INFO, DEBUG, or WARNING)
        Default is logging.INFO.
    blank_dc : bool, optional
        Remove the DC spike? (True/False)
        Default is True (recommended).
    """
    def __init__(self, datafile, max_drift=10.0, min_drift=0.00001, snr=25.0, out_dir='./', coarse_chans=None,
                 obs_info=None, flagging=False, n_coarse_chan=None, kernels=None, gpu_backend=False, gpu_id=0,
                 precision=1, append_output=False, log_level_int=logging.INFO, blank_dc=True):

        print(version_announcements)

        if not kernels:
            self.kernels = Kernels(gpu_backend, precision, gpu_id)
        else:
            self.kernels = kernels

        logger.setLevel(log_level_int)
        self.log_level_int = log_level_int

        self.min_drift = min_drift
        self.max_drift = max_drift
        self.out_dir = out_dir
        self.snr = snr

        self.data_handle = DATAHandle(datafile,
                                      out_dir=out_dir,
                                      n_coarse_chan=n_coarse_chan,
                                      coarse_chans=coarse_chans,
                                      gpu_id=gpu_id,
                                      kernels=self.kernels)

        self.n_coarse_chan = self.data_handle.n_coarse_chan

        if obs_info is None:
            obs_info = {'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0,
                        'pulsar_stats': self.kernels.np.zeros(6), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0,
                        'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}

        self.obs_info = obs_info

        self.status = True
        self.flagging = flagging
        self.append_output = append_output
        self.flag_blank_dc = blank_dc
        self.parms = 'datafile={}, max_drift={}, min_drift={}, snr={}, out_dir={}, coarse_chans={}' \
                        .format(datafile, max_drift, min_drift, snr, out_dir, coarse_chans) \
                    + ', flagging={}, n_coarse_chan={}, kernels={}, gpu_id={}, gpu_backend={}, blank_dc={}' \
                        .format(flagging, self.n_coarse_chan, kernels, gpu_id, gpu_backend, blank_dc) \
                    + ', precision={}, append_output={}, log_level_int={}, obs_info={}' \
                        .format(precision, append_output, log_level_int, obs_info)
        if (min_drift < 0) or (max_drift < 0) or (min_drift > max_drift):
            raise ValueError("Both min_drift({}) and max_drift({}) must be nonnegative.\nAlso, min_drift must be < max_drift"
                             .format(min_drift, max_drift))


    def last_logwriter(self, arg_path, arg_text):
        r'''
        Write the last LogWriter entry

        Parameters
        ----------
        arg_path : str
            Path of log for the final log entries.
        arg_text : str
            Text message to include at end of the log file.

        Returns
        -------
        None.

        '''
        logwriter = LogWriter(arg_path)
        logwriter.info(arg_text)
        logwriter.info('===== END OF LOG')


    def search(self, n_partitions=1, progress_bar='n'):
        r"""
        Top level search routine.

        Parameters
        ----------
        n_partitions : int
            Number of Dask partitions (processes) to use in parallel. Defaults to single-partition (process).
        progress_bar : str {'y', 'n'}, optional
            Enable command-line progress bar.

        Returns
        -------
        None.

        Notes
        -----
        self.data_handle.cchan_list : the list of coarse channel objects for searching,
             created by self.data_handle = DATAHandle() during __init__() execution.

        If using dask (n_partitions > 1):
        * Launch multiple drift searches in parallel.
        * Each search works on a single coarse channel object.
        * n_partitions governs the maximum number of partitions to run in parallel.
        Else, the searches are done in sequence of the coarse channel objects.

        It is not recommended to mix dask partitions with GPU mode as this could cause GPU queuing.
        """
        t0 = time.time()
        
        # Make libhdf5 errors visible.  I should not have to do this!
        unsilence_errors() # from h5py._errors

        filename_in = self.data_handle.filename
        header_in   = self.data_handle.header

        # As of 2.1.0, add max_drift_rate and obs_length to FileWriter header input
        header_in['max_drift_rate'] = self.max_drift

        wfilename = filename_in.split('/')[-1].replace('.h5', '').replace('.fits', '').replace('.fil', '')
        path_log = '{}/{}.log'.format(self.out_dir.rstrip('/'), wfilename)
        path_dat = '{}/{}.dat'.format(self.out_dir.rstrip('/'), wfilename)
        if self.append_output:
            logger.debug('Appending DAT and LOG files')
        else:
            logger.debug('Recreating DAT and LOG files')
            if os.path.exists(path_log):
                os.remove(path_log)
            if os.path.exists(path_dat):
                os.remove(path_dat)
        logwriter = LogWriter(path_log)
        filewriter = FileWriter(path_dat, header_in)

        logwriter.info(version_announcements)

        msg = "HDF5 header info: {}\n".format(self.data_handle.get_info())
        logwriter.info(msg)
        print(msg)

        msg = 'Starting ET search with parameters: ' + self.parms + '\n'
        logwriter.info(msg)
        print(msg)

        msg = "Computed drift rate resolution: {}\n".format(self.data_handle.drift_rate_resolution)
        logwriter.info(msg)
        print(msg)

        # Run serial version
        if n_partitions == 1:
            sched = Scheduler(load_the_data, [ (cchan_obj, self.kernels.precision) for cchan_obj in self.data_handle.cchan_list ])
            for cchan_obj in self.data_handle.cchan_list:
                search_coarse_channel(cchan_obj, self, dataloader=sched, filewriter=filewriter, logwriter=logwriter)
        # Run Parallel version via dask
        else:
            print("FindDoppler.search: Using {} dask partitions".format(n_partitions))
            b = db.from_sequence(self.data_handle.cchan_list, npartitions=n_partitions)
            if progress_bar == 'y':
                with ProgressBar():
                    b.map(search_coarse_channel, self).compute()
            else:
                b.map(search_coarse_channel, self).compute()
            merge_dats_logs(filename_in, self.out_dir, 'dat', cleanup='y')
            merge_dats_logs(filename_in, self.out_dir, 'log', cleanup='y')

        t1 = time.time()
        self.last_logwriter(path_log, '\n===== Search time: {:.2f} minutes'.format((t1 - t0)/60.0))

def load_the_data(cchan_dict, precision):
    r"""
    Load the DATAH5 object, spectra matrix, and the associated drift indexes.

    Parameters
    ----------
    cchan_dict : dict
        A single coarse channel object created by data_handler.py DATAHandle __split_h5.
    precision : int {2: float64, 1: float32}
        Floating point precision for the GPU.

    Returns
    -------
    datah5_obj : DATAH5 object (complex!)
    spectra : numpy.ndarray
        Spectra data array.  Set by the data_handler.py load_data function.
    drift_indexes: numpy.ndarray
        Drift index matrix.  Set by the data_handler.py load_data function.
    """
    datah5_obj = DATAH5(cchan_dict['filename'],
                  f_start=cchan_dict['f_start'],
                  f_stop=cchan_dict['f_stop'],
                  cchan_id=cchan_dict['cchan_id'],
                  n_coarse_chan=cchan_dict['n_coarse_chan'],
                  gpu_backend=False,
                  precision=precision)
    spectra, drift_indices = datah5_obj.load_data()
    datah5_obj.close()

    return (datah5_obj, spectra, drift_indices)

def search_coarse_channel(cchan_dict, fd, dataloader=None, logwriter=None, filewriter=None):
    r"""
    Run a turboseti search on a single coarse channel.

    Parameters
    ----------
    cchan_dict : dict
        A single coarse channel object created by data_handler.py DATAHandle __split_h5.
        Contains the following fields:
        * filename : file path (common to all objects)
        * f_start : start frequency of coarse channel
        * f_stop : stop frequency of coarse channel
        * cchan_id : coarse channel number
        * n_coarse_chan : total number of coarse channels (common to all objects)
    fd : FindDoppler object
        Instance of the FindDoppler class.
    logwriter : LogWriter, optional
        A LogWriter to write log output into. If None, one will be created.
    filewriter : FileWriter, optional
        A FileWriter to use to write the dat file. If None, one will be created.

    Returns
    -------
    : bool
        Returns True if no exceptions occur (needed for dask).

    Notes
    -----
    This function is separate from the FindDoppler class to allow parallelization.
    This should not be called directly, but rather via the FindDoppler.search() routine.
    One exception: turboseti_search package.
    """
    global logger

    # Fetch the identifier of this coarse channel to search.
    this_coarse_channel = cchan_dict['cchan_id']

    # Set up logging to include the coarse channel identifier.
    logger = logging.getLogger(logger_name + '.' + str(this_coarse_channel))
    logger.setLevel(fd.log_level_int)

    # Load data from file
    if dataloader:
        datah5_obj, spectra, drift_indices = dataloader.get()
    else:
        datah5_obj, spectra, drift_indices = load_the_data(cchan_dict, fd.kernels.precision)
    if this_coarse_channel == 0:
        logger.info("Spectra 0 1st 3 values: {}".format(spectra[0, 0:3]))
        logger.info("Spectra 1 1st 3 values: {}".format(spectra[1, 0:3]))

    fileroot_out = fd.data_handle.filename.split('/')[-1].replace('.h5', '').replace('.fits', '').replace('.fil', '')
    if logwriter is None:
        logwriter = LogWriter('%s/%s_%i.log' % (fd.out_dir.rstrip('/'),
                              fileroot_out,
                              this_coarse_channel))
    if filewriter is None:
        filewriter = FileWriter('%s/%s_%i.dat' % (fd.out_dir.rstrip('/'),
                                fileroot_out,
                                this_coarse_channel),
                                fd.data_handle.header)

    tsteps = datah5_obj.tsteps
    tsteps_valid = datah5_obj.tsteps_valid
    tdwidth = datah5_obj.tdwidth
    fftlen = datah5_obj.fftlen
    nframes = tsteps_valid
    shoulder_size = datah5_obj.shoulder_size

    logger.debug('===== coarse_channel={}, f_start={}, f_stop={}'
                .format(this_coarse_channel, cchan_dict['f_start'], cchan_dict['f_stop']))
    logger.debug('flagging={}, tsteps={}, tsteps_valid={}, tdwidth={}, fftlen={}, nframes={}, shoulder_size={}'
                 .format(fd.flagging, tsteps, tsteps_valid, tdwidth, fftlen, nframes, shoulder_size))

    if fd.flagging:
        ##EE This flags the edges of the PFF for BL data (with 3Hz res per channel).
        ##EE The PFF flat profile falls after around 100k channels.
        ##EE But it falls slowly enough that could use 50-80k channels.
        median_flag = fd.kernels.np.median(spectra)
        #             spectra[:,:80000] = median_flag/float(tsteps)
        #             spectra[:,-80000:] = median_flag/float(tsteps)

        ##EE Flagging spikes in time series.
        time_series = spectra.sum(axis=1)
        time_series_median = fd.kernels.np.median(time_series)

        # Flagging spikes > 10 in SNR
        mask = (time_series - time_series_median) / time_series.std() > 10
        if mask.any():
            logwriter.info("Found spikes in the time series. Removing ...")
            # So that the value is not the median in the time_series.
            spectra[mask, :] = time_series_median / float(fftlen)

    else:
        median_flag = 0
    logger.debug('median_flag={}'.format(median_flag))

    # allocate array for findopplering
    # init findopplering array to zero
    tree_findoppler = fd.kernels.xp.zeros(tsteps * tdwidth, dtype=fd.kernels.float_type) + median_flag

    # allocate array for holding original
    # Allocates array in a fast way (without initialize)
    tree_findoppler_original = fd.kernels.xp.empty_like(tree_findoppler, dtype=fd.kernels.float_type)

    # allocate array for negative doppler rates
    tree_findoppler_flip = fd.kernels.xp.empty_like(tree_findoppler, dtype=fd.kernels.float_type)

    # build index mask for in-place tree doppler correction
    ibrev = fd.kernels.np.zeros(tsteps, dtype=fd.kernels.np.int32)

    for i in range(0, tsteps):
        ibrev[i] = fd.kernels.bitrev(i, int(fd.kernels.np.log2(tsteps)))

    ##EE: should double check if tdwidth is really better than fftlen here.
    max_val = max_vals()
    if max_val.maxsnr is None:
        max_val.maxsnr = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.float_type)
    if max_val.maxdrift is None:
        max_val.maxdrift = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.float_type)
    if max_val.total_n_hits is None:
        max_val.total_n_hits = fd.kernels.xp.zeros(1, dtype=fd.kernels.xp.uint32)

    # EE: Making "shoulders" to avoid "edge effects". Could do further testing.
    specstart = int(tsteps * shoulder_size / 2)
    specend = tdwidth - (tsteps * shoulder_size)
    logger.debug('specstart={}, specend={}'.format(specstart, specend))

    # --------------------------------
    # Stats calc
    the_median, the_stddev = comp_stats(spectra.sum(axis=0), xp=fd.kernels.np)
    logger.debug('comp_stats the_median={}, the_stddev={}'.format(the_median, the_stddev))

    if fd.flag_blank_dc:
        # Remove the DC spike by making it the average of the adjacent columns
        midpoint = int(spectra.shape[1] / 2)
        spectra[:, midpoint] = (spectra[:, midpoint - 1] + spectra[:, midpoint + 1]) / 2

    # If even a line where every pixel equals the brightest point is not bright enough to produce a hit, we can
    # skip all the expensive processing of this coarse channel.
    max_point = fd.kernels.np.max(spectra)
    max_possible_line_sum = max_point * spectra.shape[0]
    max_possible_snr = (max_possible_line_sum - the_median) / the_stddev
    if max_possible_snr < fd.snr:
        logger.debug("Maximum possible SNR is %s so we can skip this coarse channel.", max_possible_snr)
        filewriter.close()
        logwriter.close()
        return True

    # --------------------------------
    # Looping over drift_rate_nblock
    # --------------------------------
    drift_rate_nblock = int(fd.kernels.np.floor(fd.max_drift / (datah5_obj.drift_rate_resolution * tsteps_valid)))
    logger.debug('BEGIN looping over drift_rate_nblock, drift_rate_nblock={}.'.format(drift_rate_nblock))

    # Transfer data to device
    spectra = fd.kernels.xp.asarray(spectra, dtype=fd.kernels.float_type)

    ##EE-debuging        kk = 0
    drift_low = -1 * drift_rate_nblock
    drift_high = drift_rate_nblock + 1
    for drift_block in range(drift_low, drift_high):
        logger.debug("Drift_block {} (in range from {} through {})"
                     .format(drift_block, drift_low, drift_rate_nblock))

        # ----------------------------------------------------------------------
        # Negative drift rates search.
        # ----------------------------------------------------------------------
        if drift_block <= 0:

            # Populates the find_doppler tree with the spectra
            populate_tree(fd, spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=1)

            # populate original array
            fd.kernels.xp.copyto(tree_findoppler_original, tree_findoppler)

            # populate neg doppler array
            fd.kernels.xp.copyto(tree_findoppler_flip, tree_findoppler_original)

            # Flip matrix across X dimension to search negative doppler drift rates
            fd.kernels.xp.copyto(tree_findoppler_flip, tree_findoppler_flip.reshape((tsteps, tdwidth))[:, ::-1].ravel())
            fd.kernels.tt.flt(tree_findoppler_flip, tsteps * tdwidth, tsteps)
            logger.debug("done...")

            complete_drift_range = datah5_obj.drift_rate_resolution * fd.kernels.np.array(
                range(-1 * tsteps_valid * (abs(drift_block) + 1) + 1,
                      -1 * tsteps_valid * (abs(drift_block)) + 1))
            bool_selected = complete_drift_range >= -fd.max_drift
            logger.debug('***** drift_block <= 0 selected drift range:\n%s', complete_drift_range[bool_selected])
            for k, drift_rate in enumerate(complete_drift_range[bool_selected]):
                # DCP 2020.04 -- WAR to drift rate in flipped files
                if datah5_obj.header['DELTAF'] < 0:
                    drift_rate = -drift_rate

                indx = ibrev[drift_indices[::-1][bool_selected][k]] * tdwidth

                # SEARCH NEGATIVE DRIFT RATES
                spectrum = tree_findoppler_flip[indx: indx + tdwidth]

                # Reverse spectrum back
                spectrum = spectrum[::-1]

                if abs(drift_rate) > fd.min_drift:
                    hitsearch(fd, spectrum, specstart, specend, fd.snr, 
                              drift_rate, datah5_obj.header,
                              tdwidth, max_val, the_median, the_stddev)

        # ----------------------------------------------------------------------
        # Positive drift rates search.
        # ----------------------------------------------------------------------
        if drift_block >= 0:

            # Populates the find_doppler tree with the spectra
            populate_tree(fd, spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=1)

            # populate original array
            fd.kernels.xp.copyto(tree_findoppler_original, tree_findoppler)
            fd.kernels.tt.flt(tree_findoppler, tsteps * tdwidth, tsteps)

            ##EE: Calculates the range of drift rates for a full drift block.
            complete_drift_range = datah5_obj.drift_rate_resolution * fd.kernels.np.array(
                range(tsteps_valid * (drift_block), tsteps_valid * (drift_block + 1)))
            bool_selected = complete_drift_range <= fd.max_drift
            logger.debug('***** drift_block >= 0 selected drift range:\n%s', complete_drift_range[bool_selected])
            for k, drift_rate in enumerate(complete_drift_range[bool_selected]):
                indx = ibrev[drift_indices[k]] * tdwidth

                # DCP 2020.04 -- WAR to drift rate in flipped files
                if datah5_obj.header['DELTAF'] < 0:
                    drift_rate = -drift_rate

                # SEARCH POSITIVE DRIFT RATES
                spectrum = tree_findoppler[indx: indx + tdwidth]

                if abs(drift_rate) > fd.min_drift:
                    hitsearch(fd, spectrum, specstart, specend, fd.snr, 
                              drift_rate, datah5_obj.header,
                              tdwidth, max_val, the_median, the_stddev)

    # Writing the top hits to file.
    logger.debug('END looping over drift_rate_nblock.')
    tophitsearch(fd, tree_findoppler_original, max_val, tsteps, datah5_obj.header, tdwidth,
                 fftlen, fd.max_drift, datah5_obj.header['obs_length'],
                 logwriter=logwriter, filewriter=filewriter, obs_info=fd.obs_info)

    logger.debug("Total number of candidates for coarse channel " +
                str(datah5_obj.header['cchan_id']) + " is: %i" % max_val.total_n_hits)
    filewriter.close()
    logwriter.close()

    return True ## NEEDED for dask


def populate_tree(fd, spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen,
                  shoulder_size, roll=0, reverse=0):
    r"""
    This script populates the findoppler tree with the spectra.

    Parameters
    ----------
    fd : FindDoppler object
        Instance of FindDoppler class.
    spectra : ndarray
        Spectra matrix.
    tree_findoppler : ndarray
        Tree to be populated with spectra.
    nframes : int
    tdwidth : int
    tsteps : int
    fftlen : int
        Length of fast fourier transform (fft) matrix.
    shoulder_size : int
        Size of shoulder region.
    roll : int, optional
        Used to calculate amount each entry to the spectra should be rolled (shifted).
    reverse : int, optional
        Used to determine which way spectra should be rolled (shifted).

    Returns
    -------
    : ndarray
        Spectra-populated version of the input tree_findoppler.

    Notes
    -----
    It creates two "shoulders" (each region of tsteps*(shoulder_size/2) in size) to avoid "edge" issues.
    It uses np.roll() for drift-rate blocks higher than 1.

    """
    logger.debug("populate_tree() roll=" + str(roll))
    if reverse:
        direction = -1
    else:
        direction = 1

    size = tsteps*int(shoulder_size/2)

    for i in range(0, nframes):
        sind_b = (i * tdwidth)
        sind_a = sind_b + size

        ##EE copy spectra into tree_findoppler, leaving two regions in each side blank (each region of tsteps*(shoulder_size/2) in size).
        # Copy spectra into tree_findoppler, with rolling.
        fd.kernels.xp.copyto(tree_findoppler[sind_a:sind_a+fftlen],
                             fd.kernels.xp.roll(spectra[i],
                             roll * i * direction))

        ##EE loads the end part of the current spectrum into the left hand side black region in tree_findoppler (comment below says "next spectra" but for that need i+1...bug?)
        #load end of current spectra into left hand side of next spectra
        fd.kernels.xp.copyto(tree_findoppler[sind_b:sind_b+size],
                             spectra[i, fftlen-size:fftlen])

    return tree_findoppler


def hitsearch(fd, spectrum, specstart, specend, snr_thresh, drift_rate,
              header, tdwidth, max_val, the_median, the_stddev):
    r"""
    Searches for hits that exceed the given SNR threshold.

    Note that the "max" arrays share the index values as any given spectrum.
    They represent maximums with respect to the frequency columns in the range (0, FFT length).

    Let S be the subspectrum given by spectrum[specstart:specend].
    Set hit-counter to 0.
    For each element of S,
        Subtract the given median and divide that result by the given standard deviation,
           giving the new element value.
        if the element value > snr_thresh then
            Increment hit-counter
            If element value > current max SNR using the common index then
                Set the current max SNR at the common index = this element.
                Set the current max drift rate at the common index  = drift rate of this element.
    Increment the grand total of hits by the hit-counter.

    Parameters
    ----------
    fd : FindDoppler
        Instance of FindDoppler class.
    spectrum : ndarray
        Array of data values along the frequency axis of length = FFT length.
    specstart : int
        First index to search for hit in spectrum.
    specend : int
        Last index to search for hit in spectrum.
    snr_thresh : float
        Minimum signal to noise ratio for candidacy.
    drift_rate : float
        Drift rate at which we are searching for hits.
    header : dict
        Header in fits header format. See data_handler.py's DATAH5 class header.
    tdwidth : int
        FFT Length = # fine channels / # coarse channels.
    max_val : max_vals
        Object to be filled with max values from this search and then returned.
        Length of each subarray = FFT length.

    """
    global logger

    logger.debug('Start searching for hits at drift rate: %f' % drift_rate)

    if fd.kernels.gpu_backend:

        blockSize = 512
        length = specend - specstart
        numBlocks = (length + blockSize - 1) // blockSize
        spectrum_tmp = fd.kernels.xp.copy(spectrum[specstart:specend])
        call = (length, spectrum_tmp, float(snr_thresh), float(drift_rate),
                max_val.maxsnr, max_val.maxdrift, max_val.total_n_hits,
                the_median, the_stddev)
        fd.kernels.hitsearch((numBlocks,), (blockSize,), call)

    else:

        # normalize
        spectrum -= the_median
        spectrum /= the_stddev

        hits = 0

        # Loop for all spectrum elements that exceed the given SNR threshold.
        # We offset each index value returned by np.nonzero()[0] by specstart
        # in order to use the returned index set on the original spectrum array.
        for i in (spectrum[specstart:specend] > snr_thresh).nonzero()[0] + specstart:

            if logger.level == logging.DEBUG:
                info_str = 'Hit found at SNR %f!\t' % (spectrum[i])
                info_str += 'Spectrum index: %d, Drift rate: %f\t' % (i, drift_rate)
                info_str += 'Uncorrected frequency: %f\t' % chan_freq(header, i, tdwidth, 0)
                info_str += 'Corrected frequency: %f' % chan_freq(header, i, tdwidth, 1)
                logger.debug(info_str)

            hits += 1
            if spectrum[i] > max_val.maxsnr[i]:
                max_val.maxsnr[i] = spectrum[i]
                max_val.maxdrift[i] = drift_rate

        max_val.total_n_hits[0] += hits

def tophitsearch(fd, tree_findoppler_original, max_val, tsteps, header, tdwidth, fftlen,
                 max_drift, obs_length, logwriter=None, filewriter=None, obs_info=None):
    r"""
    This finds the hits with largest SNR within 2*tsteps frequency channels.

    Parameters
    ----------
    tree_findoppler_original : ndarray
        Spectra-populated findoppler tree
    max_val : max_vals
        Contains max values from hitsearch
    tsteps : int
    header : dict
        Header in fits header format. Used to report tophit in filewriter.
        See :class:`~turbo_seti.find_doppler.data_handler.DATAH5`
    tdwidth : int
    fftlen : int
        Length of fast fourier transform (fft) matrix
    max_drift : float
        Maximum drift rate in Hz/second
    obs_length : float,
    logwriter : LogWriter, optional
        Logwriter to which we should write if we find a top hit.
    filewriter : FileWriter, optional
        Filewriter corresponding to file to which we should save the local maximum of tophit.
        See :func:`~turbo_seti.find_doppler.file_writers.FileWriter.report_tophit`
    obs_info : dict, optional

    Returns
    -------
    : FileWriter
        Same filewriter that was input.

    """
    global logger

    maxsnr = max_val.maxsnr
    logger.debug("original matrix size: %d\t(%d, %d)" % (len(tree_findoppler_original), tsteps, tdwidth))
    logger.debug("tree_orig shape: %s"%str((tsteps, tdwidth)))

    if fd.kernels.gpu_backend:
        maxsnr = fd.kernels.xp.asnumpy(maxsnr)

    for i in (maxsnr > 0).nonzero()[0]:
        lbound = int(max(0, i - obs_length*max_drift/2))
        ubound = int(min(tdwidth, i + obs_length*max_drift/2))

        if (maxsnr[lbound:ubound] > maxsnr[i]).nonzero()[0].any():
            logger.debug("SNR not big enough... %f pass... index: %d", maxsnr[i], i)
            continue

        drate = max_val.maxdrift[i]
        info_str = "Top hit found! SNR {:f}, Drift Rate {:f}, index {}" \
                   .format(maxsnr[i], drate.item(), i)
        logger.info(info_str)
        if logwriter:
            logwriter.info(info_str)
        if filewriter:
            filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tdwidth, fftlen, header,
                                                  max_val.total_n_hits[0], obs_info=obs_info)
        else:
            logger.error('No filewriter available in tophitsearch.')
