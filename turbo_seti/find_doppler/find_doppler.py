#!/usr/bin/env python

import importlib
import logging

# Parallel python support
import dask.bag as db
from dask.diagnostics import ProgressBar

from .kernels import Kernels
from .data_handler import DATAHandle, DATAH5
from .file_writers import FileWriter, LogWriter
from .helper_functions import chan_freq, comp_stats
from .merge_dats_logs import merge_dats_logs

#For debugging
#import pdb;# pdb.set_trace()

logger = logging.getLogger(__name__)

class max_vals:
    r"""
    Class used to initialize some maximums.

    """
    def __init__(self):
        self.maxsnr = None
        self.maxdrift = None
        self.maxsmooth = None
        self.maxid = None
        self.total_n_hits = None


class FindDoppler:
    r"""
    Initializes FindDoppler object.

    Parameters
    ----------
    datafile : string
        Inputted filename (.h5 or .fil)
    max_drift : float
        Max drift rate in Hz/second.
    min_drift : int
        Min drift rate in Hz/second.
    snr : float
        Signal to Noise Ratio - A ratio bigger than 1 to 1 has more signal than noise.
    out_dir : string
        Directory where output files should be placed. By default this is the
        current working directory.
    coarse_chans : list(int)
        The inputted comma separated list of coarse channels to analyze, if any.
    obs_info : dict
        Used to hold info found on file, including info about pulsars, RFI, and SEFD.
    flagging : bool
        Flags the edges of the PFF for BL data (with 3Hz res per channel).
    n_coarse_chan : int
        Number of coarse channels in file.
    kernels : Kernels, optional
        Pre-configured class of Kernels.
    gpu_backend : bool, optional
        Use GPU accelerated Kernels.
    precision : int {2: float64, 1: float32}, optional
        Floating point precision.

    """
    def __init__(self, datafile, max_drift, min_drift=0, snr=25.0, out_dir='./', coarse_chans=None,
                 obs_info=None, flagging=None, n_coarse_chan=None, kernels=None, gpu_backend=False,
                 precision=2):
        if not kernels:
            self.kernels = Kernels(gpu_backend, precision)
        else:
            self.kernels = kernels

        self.min_drift = min_drift
        self.max_drift = max_drift
        self.out_dir = out_dir
        self.snr = snr

        self.data_handle = DATAHandle(datafile, out_dir=out_dir, n_coarse_chan=n_coarse_chan, coarse_chans=coarse_chans, kernels=self.kernels)
        if (self.data_handle is None) or (self.data_handle.status is False):
            raise IOError("File error, aborting...")

        logger.info(self.data_handle.get_info())
        logger.info("A new FinDoppler instance created!")

        if obs_info is None:
            obs_info = {'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0,
                        'pulsar_stats': self.kernels.np.zeros(6), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0,
                        'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}

        self.obs_info = obs_info

        self.status = True
        self.flagging = flagging

    def get_info(self):
        r"""
        Generate info string.

        Returns:
          : String that contains the values of this FinDoppler object's attributes.

        """
        info_str = "File: %s\n drift rates (min, max): (%f, %f)\n SNR: %f\n"%(self.data_handle.filename, self.min_drift, self.max_drift, self.snr)
        return info_str

    def search(self, n_partitions=1, progress_bar='y'):
        r"""
        Top level search routine. 

        Parameters
        ----------
        n_partitions : int
            Number of Dask threads to use in parallel. Defaults to single-thread.
        progress_bar : str {'y', 'n'}, optional
            Enable command-line progress bar.

        Notes
        -----
        Can use dask to launch multiple drift searches in parallel.

        """
        logger.debug("Start searching...")
        logger.debug(self.get_info())

        filename_in = self.data_handle.filename
        header_in   = self.data_handle.header

        wfilename = filename_in.split('/')[-1].replace('.h5', '').replace('.fits', '').replace('.fil', '')
        logwriter = LogWriter('%s/%s.log'%(self.out_dir.rstrip('/'), wfilename))
        filewriter = FileWriter('%s/%s.dat'%(self.out_dir.rstrip('/'), wfilename), header_in)

        logger.info("Start ET search for %s" % filename_in)
        logwriter.info("Start ET search for %s" % filename_in)

        # Run serial version
        if n_partitions == 1:
            for _, data_dict in enumerate(self.data_handle.data_list):
                search_coarse_channel(data_dict, self, filewriter=filewriter, logwriter=logwriter)
        # Run Parallel version via dask
        else:
            b = db.from_sequence(self.data_handle.data_list, npartitions=n_partitions)
            if progress_bar == 'y':
                with ProgressBar():
                    b.map(search_coarse_channel, self).compute()
            else:
                b.map(search_coarse_channel, self).compute()
            merge_dats_logs(filename_in, self.out_dir, 'dat', cleanup='y')
            merge_dats_logs(filename_in, self.out_dir, 'log', cleanup='y')


def search_coarse_channel(data_dict, find_doppler_instance, logwriter=None, filewriter=None):
    r"""
    Run a turboseti search on a single coarse channel.

    Parameters
    ----------
    data_dict : dict
        File's waterfall data handler.
        Required keys: {'filename','f_start', 'f_stop', 'coarse_chan', 'n_coarse_chan'}
    find_doppler_instance : FindDoppler
        Instance of FindDoppler class.
    logwriter : LogWriter, optional
        A LogWriter to write log output into. If None, one will be created.
    filewriter : FileWriter, optional
        A FileWriter to use to write the dat file. If None, one will be created.

    Returns
    -------
    : bool
        Returns True if successful (needed for dask).

    Notes
    -----
    This function is separate from the FindDoppler class to allow parallelization. This should not be called
    directly, but rather via the `FindDoppler.search()` or `FindDoppler.search_dask()` routines.

    """
    d = data_dict
    fd = find_doppler_instance

    out_dir = fd.out_dir
    filename_in = fd.data_handle.filename
    header_in = fd.data_handle.header
    min_drift = fd.min_drift
    max_drift = fd.max_drift
    snr = fd.snr
    obs_info = fd.obs_info
    flagging = fd.flagging

    #logger.info("Start searching for coarse channel: %s" % d['coarse_chan'])
    data_obj = DATAH5(d['filename'], f_start=d['f_start'], f_stop=d['f_stop'],
                      coarse_chan=d['coarse_chan'], n_coarse_chan=d['n_coarse_chan'], kernels=fd.kernels)

    fileroot_out = filename_in.split('/')[-1].replace('.h5', '').replace('.fits', '').replace('.fil', '')
    if logwriter is None:
        logwriter = LogWriter('%s/%s_%i.log' % (out_dir.rstrip('/'), fileroot_out, d['coarse_chan']))
    if filewriter is None:
        filewriter = FileWriter('%s/%s_%i.dat' % (out_dir.rstrip('/'), fileroot_out, d['coarse_chan']), header_in)

    spectra, drift_indices = data_obj.load_data()
    spectra_flipped = fd.kernels.xp.copy(spectra)[:, ::-1]
    tsteps = data_obj.tsteps
    tsteps_valid = data_obj.tsteps_valid
    tdwidth = data_obj.tdwidth
    fftlen = data_obj.fftlen
    nframes = tsteps_valid
    shoulder_size = data_obj.shoulder_size

    if flagging:
        ##EE This flags the edges of the PFF for BL data (with 3Hz res per channel).
        ##EE The PFF flat profile falls after around 100k channels.
        ##EE But it falls slowly enough that could use 50-80k channels.
        median_flag = fd.kernels.xp.median(spectra)
        #             spectra[:,:80000] = median_flag/float(tsteps)
        #             spectra[:,-80000:] = median_flag/float(tsteps)

        ##EE Flagging spikes in time series.
        time_series = spectra.sum(axis=1)
        time_series_median = fd.kernels.xp.median(time_series)

        # Flagging spikes > 10 in SNR
        mask = (time_series - time_series_median) / time_series.std() > 10
        if mask.any():
            logwriter.info("Found spikes in the time series. Removing ...")
            # So that the value is not the median in the time_series.
            spectra[mask, :] = time_series_median / float(fftlen)

    else:
        median_flag = fd.kernels.xp.array([0], dtype=fd.kernels.float_type)

    # allocate array for findopplering
    # init findopplering array to zero
    tree_findoppler = fd.kernels.xp.zeros(tsteps * tdwidth, dtype=fd.kernels.float_type) + median_flag

    # allocate array for holding original
    # Allocates array in a fast way (without initialize)
    tree_findoppler_original = fd.kernels.xp.empty_like(tree_findoppler, dtype=fd.kernels.float_type)

    hitsearch_buf = fd.kernels.xp.empty(tdwidth, dtype=fd.kernels.float_type)

    # build index mask for in-place tree doppler correction
    ibrev = fd.kernels.xp.zeros(tsteps, dtype=fd.kernels.xp.int32)

    for i in range(0, tsteps):
        ibrev[i] = fd.kernels.bitrev(i, int(fd.kernels.np.log2(tsteps)))

    ##EE: should double check if tdwidth is really better than fftlen here.
    max_val = max_vals()
    if max_val.maxsnr is None:
        max_val.maxsnr = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.float_type)
    if max_val.maxdrift is None:
        max_val.maxdrift = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.float_type)
    if max_val.maxsmooth is None:
        max_val.maxsmooth = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.xp.uint8)
    if max_val.maxid is None:
        max_val.maxid = fd.kernels.xp.zeros(tdwidth, dtype=fd.kernels.xp.uint32)
    if max_val.total_n_hits is None:
        max_val.total_n_hits = fd.kernels.xp.zeros(1, dtype=fd.kernels.xp.uint32)

    # EE: Making "shoulders" to avoid "edge effects". Could do further testing.
    specstart = int(tsteps * shoulder_size / 2)
    specend = tdwidth - (tsteps * shoulder_size)

    # --------------------------------
    # Stats calc
    the_mean_val, the_stddev = comp_stats(spectra.sum(axis=0), xp=fd.kernels.xp)

    # --------------------------------
    # Looping over drift_rate_nblock
    # --------------------------------
    drift_rate_nblock = int(fd.kernels.xp.floor(max_drift / (data_obj.drift_rate_resolution * tsteps_valid)))

    ##EE-debuging        kk = 0

    for drift_block in range(-1 * drift_rate_nblock, drift_rate_nblock + 1):
        logger.debug("Drift_block %i" % drift_block)

        if drift_block < 0:
            populate_tree(fd, spectra_flipped, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=0)
        else:
            populate_tree(fd, spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=1)

        if drift_block == drift_rate_nblock:
            fd.kernels.xp.copyto(tree_findoppler_original, tree_findoppler)
        fd.kernels.tt.flt(tree_findoppler, tsteps * tdwidth, tsteps)

        if drift_block < 0:
            logger.info("Un-flipping corrected negative doppler...")
            tree_findoppler = tree_findoppler[::-1]

        tree_findoppler -= the_mean_val
        tree_findoppler /= the_stddev

        if drift_block < 0:
            complete_drift_range = data_obj.drift_rate_resolution * fd.kernels.np.array(
                range(-1 * tsteps_valid * (abs(drift_block) + 1) + 1,
                      -1 * tsteps_valid * (abs(drift_block)) + 1))
            sub_range = complete_drift_range[(complete_drift_range < min_drift) &
                                             (complete_drift_range >= -1 * max_drift)]
        else:
            complete_drift_range = data_obj.drift_rate_resolution * fd.kernels.np.array(
                range(tsteps_valid * drift_block, tsteps_valid * (drift_block + 1)))
            sub_range = complete_drift_range[(complete_drift_range >= min_drift) &
                                             (complete_drift_range <= max_drift)]

        for k, drift_rate in enumerate(sub_range):
            # DCP 2020.04 -- WAR to drift rate in flipped files
            if data_obj.header['DELTAF'] < 0:
                drift_rate *= -1

            # Grab correct bit of spectrum out of the dedoppler tree output
            indx = ibrev[drift_indices[k]] * tdwidth
            fd.kernels.xp.copyto(hitsearch_buf, tree_findoppler[indx: indx + tdwidth])
            hitsearch(fd, hitsearch_buf, specstart, specend, snr,
                      drift_rate, data_obj.header, tdwidth, max_val, 0)

    # Writing the top hits to file.
    filewriter = tophitsearch(tree_findoppler_original, max_val, tsteps, data_obj.header, tdwidth,
                              fftlen, max_drift, data_obj.obs_length,
                              logwriter=logwriter, filewriter=filewriter, obs_info=obs_info)

    logger.info("Total number of candidates for coarse channel " + 
                str(data_obj.header['coarse_chan']) + " is: %i" % max_val.total_n_hits)
    data_obj.close()
    filewriter.close()
    return True


def populate_tree(fd, spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen,
                  shoulder_size, roll=0, reverse=0):
    r"""
    This script populates the findoppler tree with the spectra.

    Parameters
    ----------
    fd : FindDoppler
        Instance of FindDoppler class.
    spectra : ndarray
        Spectra calculated from file.
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
    if reverse:
        direction = -1
    else:
        direction = 1

    size = tsteps*int(shoulder_size/2)

    for i in range(0, nframes):
        sind_b = (i * tdwidth)
        sind_a = sind_b + size

        ##EE copy spectra into tree_findoppler, leaving two regions in each side blanck (each region of tsteps*(shoulder_size/2) in size).
        # Copy spectra into tree_findoppler, with rolling.
        fd.kernels.xp.copyto(tree_findoppler[sind_a:sind_a+fftlen], fd.kernels.xp.roll(spectra[i], roll * i * direction))

        ##EE loads the end part of the current spectrum into the left hand side black region in tree_findoppler (comment below says "next spectra" but for that need i+1...bug?)
        #load end of current spectra into left hand side of next spectra
        fd.kernels.xp.copyto(tree_findoppler[sind_b:sind_b+size], spectra[i, fftlen-size:fftlen])

    return tree_findoppler


def hitsearch(fd, spectrum, specstart, specend, hitthresh, drift_rate, header, tdwidth, max_val, reverse):
    r"""
    Searches for hits at given drift rate. A hit occurs in each channel if > hitthresh.

    Parameters
    ----------
    fd : FindDoppler
        Instance of FindDoppler class.
    spectrum : ndarray
    specstart : int
        First index to search for hit in spectrum.
    specend : int
        Last index to search for hit in spectrum.
    hitthresh : float
        Signal to noise ratio used as threshold for determining hits.
    drift_rate : float
        Drift rate at which we are searching for hits.
    header : dict
        Header in fits header format. See data_handler.py's DATAH5 class header.
    tdwidth : int
    max_val : max_vals
        Object to be filled with max values from this search and then returned.
    reverse : int
        Used to flag whether fine channel should be reversed.

    """
    logger.debug('Start searching for hits at drift rate: %f' % drift_rate)

    if fd.kernels.gpu_backend:

        blockSize = 512
        length = specend - specstart
        numBlocks = (length + blockSize - 1) // blockSize
        call = (length, spectrum[specstart:specend], hitthresh, drift_rate,
                max_val.maxsnr, max_val.maxdrift, max_val.total_n_hits)
        fd.kernels.hitsearch((numBlocks,), (blockSize,), call)

    else:

        hits = 0
        for i in (spectrum[specstart:specend] > hitthresh).nonzero()[0] + specstart:
            k = (tdwidth - 1 - i) if reverse else i

            if logger.level == logging.DEBUG:
                info_str = 'Hit found at SNR %f! %s\t' % (spectrum[i], '(reverse)' if reverse else '')
                info_str += 'Spectrum index: %d, Drift rate: %f\t' % (i, drift_rate)
                info_str += 'Uncorrected frequency: %f\t' % chan_freq(header, k, tdwidth, 0)
                info_str += 'Corrected frequency: %f' % chan_freq(header, k, tdwidth, 1)
                logger.debug(info_str)

            hits += 1
            if spectrum[i] > max_val.maxsnr[k]:
                max_val.maxsnr[k] = spectrum[i]
                max_val.maxdrift[k] = drift_rate
                max_val.maxid[k] = hits

        max_val.total_n_hits[0] += hits


def tophitsearch(tree_findoppler_original, max_val, tsteps, header, tdwidth, fftlen,
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
        Max drift rate in Hz/second
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
    maxsnr = max_val.maxsnr
    logger.debug("original matrix size: %d\t(%d, %d)"%(len(tree_findoppler_original), tsteps, tdwidth))
    tree_orig = tree_findoppler_original.reshape((tsteps, tdwidth))
    logger.debug("tree_orig shape: %s"%str(tree_orig.shape))

    for i in (maxsnr > 0).nonzero()[0]:
        lbound = int(max(0, i - obs_length*max_drift/2))
        ubound = int(min(tdwidth, i + obs_length*max_drift/2))

        skip = 0

        if (maxsnr[lbound:ubound] > maxsnr[i]).nonzero()[0].any():
            skip = 1

        if skip:
            logger.debug("SNR not big enough... %f pass... index: %d"%(maxsnr[i], i))
        else:
            info_str = "Top hit found! SNR: %f ... index: %d"%(maxsnr[i], i)
            logger.info(info_str)
            if logwriter:
                logwriter.info(info_str)
            if filewriter:
                filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tdwidth, fftlen, header,max_val.total_n_hits[0], obs_info=obs_info)
            else:
                logger.error('Not have filewriter? tell me why.')

    return filewriter
