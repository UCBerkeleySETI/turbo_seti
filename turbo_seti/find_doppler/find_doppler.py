"""
Class FindDoppler

Non-object functions:
    search_coarse_channel
    populate_tree
    hitsearch
    tophitsearch
"""

# Parallel python support
import dask.bag as db
from dask.diagnostics import ProgressBar

import numpy as np
import copy

#For importing cython code
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from .data_handler import DATAHandle, DATAH5
from .file_writers import FileWriter, LogWriter
from .helper_functions import bitrev, chan_freq, comp_stats, FlipX
from .merge_dats_logs import merge_dats_logs

try:
    from . import taylor_tree as tt
except:
    import taylor_tree as tt

#For debugging
#import pdb;# pdb.set_trace()

from logbook import Logger
from ..log import logger_group
logger = Logger('turboseti.find_doppler')
logger_group.add_logger(logger)

class MaxVals:
    """ Class used to initialize some maximums """
    def __init__(self):
        self.maxsnr = None
        self.maxdrift = None
        self.maxsmooth = None
        self.maxid = None
        self.total_n_hits = None

    def init(self, tdwidth):
        if self.maxsnr is None:
            self.maxsnr = np.zeros(tdwidth, dtype=np.float64)
        if self.maxdrift is None:
            self.maxdrift = np.zeros(tdwidth, dtype=np.float64)
        if self.maxsmooth is None:
            self.maxsmooth = np.zeros(tdwidth, dtype='uint8')
        if self.maxid is None:
            self.maxid = np.zeros(tdwidth, dtype='uint32')
        if self.total_n_hits is None:
            self.total_n_hits = 0

class HistVals:
    """Temporary class that saved the normalized spectrum for all drift rates."""
    def __init__(self):
        self.histsnr = None
        self.histdrift = None
        self.histid = None


class FindDoppler:
    """ """
    def __init__(self, datafile, max_drift, min_drift=0, snr=25.0, out_dir='./', coarse_chans=None,
                 obs_info=None, flagging=None, n_coarse_chan=None, fscrunch=1):
        """
        Initializes FindDoppler object

        Args:
            datafile (string):  Inputted filename (.h5 or .fil)
            max_drift (float):  Max drift rate in Hz/second
            min_drift (int):    Min drift rate in Hz/second
            snr (float):        Signal to Noise Ratio - A ratio bigger than 1 to 1 has more signal than  noise
            out_dir (string):   Directory where output files should be placed. By default this is the
                                current working directory.
            coarse_chans (list(int)):  the inputted comma separated list of coarse channels to analyze, if any.
            obs_info (dict):     Used to hold info found on file, including info about pulsars, RFI, and SEFD
            flagging (bool):     Flags the edges of the PFF for BL data (with 3Hz res per channel)
            n_coarse_chan (int): Number of coarse channels in file
        """
        self.min_drift = min_drift
        self.max_drift = max_drift
        self.snr = snr
        self.out_dir = out_dir
        self.fscrunch = fscrunch

        self.data_handle = DATAHandle(datafile, out_dir=out_dir, n_coarse_chan=n_coarse_chan, coarse_chans=coarse_chans)
        if (self.data_handle is None) or (self.data_handle.status is False):
            raise IOError("File error, aborting...")

        logger.info(self.data_handle.get_info())
        logger.info("A new FinDoppler instance created!")

        if obs_info is None:
            obs_info = {'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0,
                        'pulsar_stats': np.zeros(6), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0,
                        'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}

        self.obs_info = obs_info

        self.status = True
        self.flagging = flagging

    def get_info(self):
        """Generate info string

        Args: None

        Returns:
          : String that contains the values of this FinDoppler object's attributes.

        """
        info_str = "File: %s\n drift rates (min, max): (%f, %f)\n SNR: %f\n"%(self.data_handle.filename, self.min_drift, self.max_drift, self.snr)
        return info_str

    def search(self, n_partitions=1, progress_bar='y'):
        """ Top level search routine

        Uses dask to launch multiple drift searches in parallel

        Args:
            n_partitions (int): Number of threads to use in parallel. Defaults to 1 (single-threaded)
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
            for ii, data_dict in enumerate(self.data_handle.data_list):
                search_coarse_channel(data_dict, self, fscrunch=self.fscrunch, filewriter=filewriter, logwriter=logwriter)
        # Run Parallel version via dask
        else:
            b = db.from_sequence(self.data_handle.data_list, npartitions=n_partitions)
            if progress_bar == 'y':
                with ProgressBar():
                    b.map(search_coarse_channel, self, fscrunch=self.fscrunch).compute()
            else:
                b.map(search_coarse_channel, self, fscrunch=self.fscrunch).compute()
                
            merge_dats_logs(filename_in, self.out_dir, 'dat', cleanup='y')
            merge_dats_logs(filename_in, self.out_dir, 'log', cleanup='y')



def search_coarse_channel(data_dict, find_doppler_instance, fscrunch=1, logwriter=None, filewriter=None):
    """ Run a turboseti search on a single coarse channel.

    This function is separate from the FindDoppler class to allow parallelization. This should not be called
    directly, but rather via the `FindDoppler.search()` or `FindDoppler.search_dask()` routines.

    Args:
        data_dict (dict): File's waterfall data handler.
                        Required keys: {'filename','f_start', 'f_stop', 'coarse_chan', 'n_coarse_chan'}
        find_doppler_instance (FindDoppler): Instance of FindDoppler class (needed to access search params)
        logwriter (LogWriter): A LogWriter to write log output into. If None, one will be created.
        filewriter (FileWriter): A FileWriter to use to write the dat file. If None, one will be created.
    Returns:
        Success (bool): Returns True if successful (needed for dask).
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
                      coarse_chan=d['coarse_chan'], n_coarse_chan=d['n_coarse_chan'])

    fileroot_out = filename_in.split('/')[-1].replace('.h5', '').replace('.fits', '').replace('.fil', '')
    if logwriter is None:
        logwriter = LogWriter('%s/%s_%i.log' % (out_dir.rstrip('/'), fileroot_out, d['coarse_chan']))
    if filewriter is None:
        filewriter = FileWriter('%s/%s_%i.dat' % (out_dir.rstrip('/'), fileroot_out, d['coarse_chan']), header_in)

    spectra, drift_indices = data_obj.load_data()
    spectra_flipped = np.copy(spectra)[:, ::-1]
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
        median_flag = np.median(spectra)
        #             spectra[:,:80000] = median_flag/float(tsteps)
        #             spectra[:,-80000:] = median_flag/float(tsteps)

        ##EE Flagging spikes in time series.
        time_series = spectra.sum(axis=1)
        time_series_median = np.median(time_series)

        # Flagging spikes > 10 in SNR
        mask = (time_series - time_series_median) / time_series.std() > 10
        if mask.any():
            logwriter.info("Found spikes in the time series. Removing ...")
            spectra[mask, :] = time_series_median / float(
                fftlen)  # So that the value is not the median in the time_series.

    else:
        median_flag = np.array([0])

    # allocate array for findopplering
    # init findopplering array to zero
    tree_findoppler = np.zeros(tsteps * tdwidth, dtype=np.float64) + median_flag

    # allocate array for holding original
    # Allocates array in a fast way (without initialize)
    tree_findoppler_original = np.empty_like(tree_findoppler)

    # build index mask for in-place tree doppler correction
    ibrev = np.zeros(tsteps, dtype=np.int32)

    for i in range(0, tsteps):
        ibrev[i] = bitrev(i, int(np.log2(tsteps)))

    d_max_val = {}
    fs = 1
    while fs <= fscrunch:
        max_val = MaxVals()
        max_val.init(tdwidth // fs)
        d_max_val[fs] = max_val
        fs *= 2

    # EE: Making "shoulders" to avoid "edge effects". Could do further testing.
    specstart = int(tsteps * shoulder_size / 2)
    specend = tdwidth - (tsteps * shoulder_size)

    # --------------------------------
    # Stats calc
    the_mean_val, the_stddev = comp_stats(spectra.sum(axis=0))

    # --------------------------------
    # Looping over drift_rate_nblock
    # --------------------------------
    drift_rate_nblock = int(np.floor(max_drift / (data_obj.drift_rate_resolution * tsteps_valid)))

    for drift_block in range(-1 * drift_rate_nblock, drift_rate_nblock + 1):
        logger.debug("Drift_block %i" % drift_block)

        # Populates the find_doppler tree with the spectra
        if drift_block < 0:
            populate_tree(spectra_flipped, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=0)
        else:
            populate_tree(spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen, shoulder_size,
                          roll=drift_block, reverse=1)

        # make a copy of original array (BEFORE TAYLOR_FLT)
        np.copyto(tree_findoppler_original, tree_findoppler)

        # Run taylor tree (main turboseti kernel)
        logger.info("Doppler correcting block %i..." % drift_block)
        tt.taylor_flt(tree_findoppler, tsteps * tdwidth, tsteps)
        logger.debug("done...")

        if drift_block < 0:
            logger.info("Un-flipping corrected negative doppler...")
            tree_findoppler = tree_findoppler[::-1]

        # DCP: Well this is horrible code:
        if drift_block < 0:
            complete_drift_range = data_obj.drift_rate_resolution * np.array(
                range(-1 * tsteps_valid * (np.abs(drift_block) + 1) + 1,
                      -1 * tsteps_valid * (np.abs(drift_block)) + 1))
            sub_range = complete_drift_range[(complete_drift_range < min_drift) &
                                             (complete_drift_range >= -1 * max_drift)]
        else:
            complete_drift_range = data_obj.drift_rate_resolution * np.array(
                range(tsteps_valid * drift_block, tsteps_valid * (drift_block + 1)))
            sub_range = complete_drift_range[(complete_drift_range >= min_drift) &
                                             (complete_drift_range <= max_drift)]

        # Loop over drift rate, and find max SNR values.
        # The MaxVals() dict, d_max_val, will update with highest SNR found.
        for k, drift_rate in enumerate(sub_range):

            # DCP 2020.04 -- WAR to drift rate in flipped files
            if data_obj.header['DELTAF'] < 0:
                drift_rate *= -1

            # Grab correct bit of spectrum out of the dedoppler tree output
            indx = ibrev[drift_indices[k]] * tdwidth
            spectrum = tree_findoppler[indx: indx + tdwidth]

            # normalize DCP: CAN THIS BE MOVED OUT OF LOOP?
            spectrum -= the_mean_val
            spectrum /= the_stddev

            n_hits, d_max_val = hitsearch(spectrum, specstart, specend, snr,
                                          drift_rate, data_obj.header,
                                          tdwidth, d_max_val, 0, fscrunch=fscrunch)

    # Writing the top hits to file.
    filewriter = tophitsearch(tree_findoppler_original, d_max_val, tsteps, data_obj.header, tdwidth,
                              fftlen, max_drift, data_obj.obs_length,
                              logwriter=logwriter, filewriter=filewriter, obs_info=obs_info, fscrunch=fscrunch)

    logger.info("Total number of candidates for coarse channel " + str(
        data_obj.header['coarse_chan']) + " is: %i" % d_max_val[1].total_n_hits)
    data_obj.close()
    filewriter.close()
    return True


def populate_tree(spectra, tree_findoppler, nframes, tdwidth, tsteps, fftlen,
                  shoulder_size, roll=0, reverse=0):
    """This script populates the findoppler tree with the spectra.
    It creates two "shoulders" (each region of tsteps*(shoulder_size/2) in size) to avoid "edge" issues.
    It uses np.roll() for drift-rate blocks higher than 1.

    Args:
      spectra: ndarray,        spectra calculated from file
      tree_findoppler: ndarray,        tree to be populated with spectra
      nframes: int,
      tdwidth: int,
      tsteps: int,
      fftlen: int,            length of fast fourier transform (fft) matrix
      shoulder_size: int,            size of shoulder region
      roll: int,            used to calculate amount each entry to the spectra should be rolled (shifted) (Default value = 0)
      reverse: int(boolean),   used to determine which way spectra should be rolled (shifted) (Default value = 0)

    Returns:
      : ndarray,        spectra-populated version of the input tree_findoppler

    """

    if reverse:
        direction = -1
    else:
        direction = 1

    for i in range(0, nframes):
        sind = tdwidth*i + tsteps*int(shoulder_size/2)
        cplen = fftlen

        ##EE copy spectra into tree_findoppler, leaving two regions in each side blanck (each region of tsteps*(shoulder_size/2) in size).
        # Copy spectra into tree_findoppler, with rolling.
        np.copyto(tree_findoppler[sind: sind + cplen], np.roll(spectra[i], roll * i * direction))

#         ##EE loads the end part of the current spectrum into the left hand side black region in tree_findoppler (comment below says "next spectra" but for that need i+1...bug?)
         #load end of current spectra into left hand side of next spectra
        sind = i * tdwidth
        np.copyto(tree_findoppler[sind: sind + tsteps*int(shoulder_size/2)], spectra[i, fftlen-(tsteps*int(shoulder_size/2)):fftlen])

    return tree_findoppler


def hitsearch(spectrum, specstart, specend, hitthresh, drift_rate, header,
              tdwidth, d_max_val, reverse, fscrunch=1):
    """Searches for hits at given drift rate. A hit occurs in each channel if > hitthresh.

    Args:
      spectrum: ndarray,
      specstart: int,                first index to search for hit in spectrum
      specend: int,                last index to search for hit in spectrum
      hitthresh: float,              signal to noise ratio used as threshold for determining hits
      drift_rate: float,              drift rate at which we are searching for hits
      header: dict,               header in fits header format. See data_handler.py's DATAH5 class header
      tdwidth: int,
      max_val: max_vals,           object to be filled with max values from this search and then returned
      reverse: int(boolean),       used to flag whether fine channel should be reversed
      fscrunch (int):       Apply frequency scrunch
    Returns:
      : int, max_vals,      j is the amount of hits.

    """
    fs = 1
    d_n_hits = {}

    # Make sure we don't modify header outside this method!
    header = copy.deepcopy(header)

    while fs <= fscrunch:
        max_val = d_max_val[fs]

        logstr = 'Start searching for hits at drift rate: %f, fs: %s' % (drift_rate, fs)
        logger.debug(logstr)
        sq2 = np.sqrt(2)
        fs2 = np.log2(fs)
        snrcorr = 1.0/np.power(sq2, fs2)  # Correction for SNR after fscrunch
        j = 0
        if fs > 1:
            spectrum  = spectrum.reshape((-1, 2)).sum(axis=-1)
            specstart = round(specstart / 2)
            specend   = round(specend / 2)
            tdwidth   = tdwidth // 2
            hitthresh *= np.sqrt(2)  # To avoid recomputing SNR/STDEV of data, scale threshold
            header['NAXIS1'] = header['NAXIS1'] // 2
            header['DELTAF'] = header['DELTAF'] * 2
            #logger.debug(spectrum.shape, specstart, specend)

        for i in (spectrum[specstart:specend] > hitthresh).nonzero()[0] + specstart:
            k = (tdwidth - 1 - i) if reverse else i
            info_str  = 'Hit found at SNR %f! %s\t' % (spectrum[i] * snrcorr, '(reverse)' if reverse else '')
            info_str += 'Spectrum index: %d, Drift rate: %f\t' % (i, drift_rate)
            info_str += 'Uncorrected frequency: %f\t' % chan_freq(header, k, tdwidth, 0)
            #info_str += 'Corrected frequency: %f' % chan_freq(header, k, tdwidth, 1)
            logger.debug(info_str)
            j += 1
            used_id = j
            if spectrum[i] * snrcorr > max_val.maxsnr[k]:
                max_val.maxsnr[k] = spectrum[i] * snrcorr
                max_val.maxdrift[k] = drift_rate
                max_val.maxid[k] = used_id
        max_val.total_n_hits += j
        d_max_val[fs] = max_val
        d_n_hits[fs] = j
        fs *= 2

    return d_n_hits, d_max_val


def tophitsearch(tree_findoppler_original, d_max_val, tsteps, header, tdwidth, fftlen,
                 max_drift, obs_length, logwriter=None, filewriter=None, obs_info=None, fscrunch=1):
    """This finds the hits with largest SNR within 2*tsteps frequency channels.

    Args:
      tree_findoppler_original: ndarray,        spectra-populated findoppler tree
      d_max_val: max_vals,      dict contains max values from hitsearch
      tsteps: int,
      header: dict,           header in fits header format. See data_handler.py's DATAH5 class header. Used to report tophit in filewriter
      tdwidth: int,
      fftlen: int,            length of fast fourier transform (fft) matrix
      max_drift: float,          Max drift rate in Hz/second
      obs_length: float,
      logwriter: LogWriter       logwriter to which we should write if we find a top hit (Default value = None)
      filewriter: FileWriter      filewriter corresponding to file to which we should save the
    local maximum of tophit. See report_tophit in filewriters.py (Default value = None)
      obs_info: dict, (Default value = None)

    Returns:
      : FileWriter,     same filewriter that was input

    """
    fs = 1
    header = copy.deepcopy(header)
    tdwidth0 = copy.copy(tdwidth)
    fftlen0 = fftlen
    deltaf0 = header['DELTAF']
    header0 = copy.deepcopy(header)

    while fs <= fscrunch:
        max_val = d_max_val[fs]
        maxsnr = max_val.maxsnr

        if fs > 1:
            logger.debug("TOPHITSEARCH fscrunching, ", fs)
            tree_findoppler_original = tree_findoppler_original.reshape((-1, 2)).mean(axis=-1)
            tdwidth = tdwidth // 2
            fftlen  = fftlen // 2
            header['NAXIS1'] = header['NAXIS1'] // 2
            header['DELTAF'] = header['DELTAF'] * 2
            logger.info("%s %s %s" %(header['NAXIS1'], header['DELTAF'], tdwidth))

        total_drift_in_hz  = obs_length * max_drift
        total_drift_n_chan = abs(round(total_drift_in_hz / header0['DELTAF'] / 1e6 / fs))
        #print(total_drift_n_chan)

        logger.info(f"max drift {max_drift}  obslen {obs_length:2.2f} tdrift {total_drift_in_hz:2.3f} tdrift_nchan {total_drift_n_chan} fscrunch {fs}")

        logger.info("original matrix size: %d\t(%d, %d)"%(len(tree_findoppler_original), tsteps, tdwidth))
        tree_orig = tree_findoppler_original.reshape((tsteps, tdwidth))
        logger.info("tree_orig shape: %s"%str(tree_orig.shape))

        for i in (maxsnr > 0).nonzero()[0]:
            lbound = int(round(max(0, i - total_drift_n_chan)))
            ubound = int(round(min(tdwidth, i + total_drift_n_chan)))
            logger.debug(f"lb:ub {lbound}:{ubound} nchan: {tdwidth} {total_drift_n_chan}")
            skip = 0

            if (maxsnr[lbound:ubound] > maxsnr[i]).nonzero()[0].any():
                skip = 1
            if skip:
                logger.debug("SNR not big enough... %f pass... index: %d"%(maxsnr[i], i))
            else:
                info_str = "Top hit found! SNR: %2.1f ... index: %d"%(maxsnr[i], i)
                logger.info(info_str)
                if logwriter:
                    logwriter.info(info_str)
                if filewriter:
                    filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tdwidth0, fftlen0, header0, max_val.total_n_hits, fs, obs_info=obs_info)
                else:
                    logger.error('Not have filewriter? tell me why.')
        fs *= 2
    return filewriter
