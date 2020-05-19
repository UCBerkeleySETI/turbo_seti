#!/usr/bin/env python


import numpy as np
import sys
import os
import logging
logger = logging.getLogger(__name__)
import gc   #Garbage collector.

from .data_handler import DATAHandle
from .file_writers import FileWriter, LogWriter
from .helper_functions import *

#For importing cython code
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

try:
    from . import taylor_tree as tt
except:
    import taylor_tree as tt

#For debugging
#import pdb;# pdb.set_trace()
class max_vals:
    """ """
    def __init__(self):
        self.maxsnr = None
        self.maxdrift = None
        self.maxsmooth = None
        self.maxid = None
        self.total_n_hits = None

class hist_vals:
    """Temporary class that saved the normalized spectrum for all drift rates."""
    def __init__(self):
        self.histsnr = None
        self.histdrift = None
        self.histid = None

class FindDoppler:
    """ """
    def __init__(self, datafile, max_drift, min_drift=0, snr=25.0, out_dir='./', coarse_chans=None, obs_info=None, flagging=None, n_coarse_chan=None):
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

        self.data_handle = DATAHandle(datafile, out_dir=out_dir, n_coarse_chan=n_coarse_chan, coarse_chans=coarse_chans)
        if (self.data_handle is None) or (self.data_handle.status is False):
            raise IOError("File error, aborting...")

        logger.info(self.data_handle.get_info())
        logger.info("A new FinDoppler instance created!")

        if obs_info is None:
            obs_info = {}
            obs_info['pulsar']        = 0  # Bool if pulsar detection.
            obs_info['pulsar_found']  = 0  # Bool if pulsar detection.
            obs_info['pulsar_dm']     = 0.0  # Pulsar expected DM.
            obs_info['pulsar_snr']    = 0.0  # Signal toNoise Ratio (SNR)
            obs_info['pulsar_stats']  = np.zeros(6)
            obs_info['RFI_level']     = 0.0  # Radio Frequency Interference
            obs_info['Mean_SEFD']     = 0.0  # Mean System Equivalent Flux Density
            obs_info['psrflux_Sens']  = 0.0
            obs_info['SEFDs_val']     = [0.0]  # System Equivalent Flux Density values
            obs_info['SEFDs_freq']    = [0.0]  # System Equivalent Flux Density frequency
            obs_info['SEFDs_freq_up'] = [0.0]

        self.obs_info = obs_info

        self.status = True
        self.flagging = flagging

    def get_info(self):
        """Generate info string

        Args:

        Returns:
          : String that contains the values of this FinDoppler object's attributes.

        """
        info_str = "File: %s\n drift rates (min, max): (%f, %f)\n SNR: %f\n"%(self.data_handle.filename, self.min_drift, self.max_drift,self.snr)
        return info_str

    def search(self):
        """Top level search routine"""
        logger.debug("Start searching...")
        logger.debug(self.get_info())

        self.logwriter = LogWriter('%s/%s.log'%(self.out_dir.rstrip('/'),
                                    self.data_handle.data_list[0].filename.split('/')[-1].replace('.h5','').replace('.fits','').replace('.fil','')))
        self.filewriter = FileWriter('%s/%s.dat'%(self.out_dir.rstrip('/'),
                                     self.data_handle.data_list[0].filename.split('/')[-1].replace('.h5','').replace('.fits','').replace('.fil','')),
                                     self.data_handle.data_list[0].header)

        logger.info("Start ET search for %s"%self.data_handle.data_list[0].filename)
        self.logwriter.info("Start ET search for %s"%(self.data_handle.data_list[0].filename))

        for ii,target_data_obj in enumerate(self.data_handle.data_list):
            self.search_data(target_data_obj)
            self.data_handle.data_list[ii].close()
            gc.collect()

    def search_data(self, data_obj):
        """Search the waterfall data of a data handler (coarse channel).

        Args:
          data_obj(DATAH5): File's waterfall data handler

        Returns:

        """
        logger.info("Start searching for coarse channel: %s"%data_obj.header['coarse_chan'])
        self.logwriter.info("Start searching for %s ; coarse channel: %i "%(data_obj.filename,data_obj.header['coarse_chan']))
        spectra, drift_indices = data_obj.load_data()
        tsteps = data_obj.tsteps
        tsteps_valid = data_obj.tsteps_valid
        tdwidth = data_obj.tdwidth
        fftlen = data_obj.fftlen
        nframes = tsteps_valid
        shoulder_size = data_obj.shoulder_size

        if self.flagging:
            ##EE This flags the edges of the PFF for BL data (with 3Hz res per channel).
            ##EE The PFF flat profile falls after around 100k channels.
            ##EE But it falls slowly enough that could use 50-80k channels.
            median_flag = np.median(spectra)
#             spectra[:,:80000] = median_flag/float(tsteps)
#             spectra[:,-80000:] = median_flag/float(tsteps)

            ##EE Flagging spikes in time series.
            time_series=spectra.sum(axis=1)
            time_series_median = np.median(time_series)
            mask=(time_series-time_series_median)/time_series.std() > 10   #Flagging spikes > 10 in SNR

            if mask.any():
                self.logwriter.info("Found spikes in the time series. Removing ...")
                spectra[mask,:] = time_series_median/float(fftlen)  # So that the value is not the median in the time_series.

        else:
            median_flag = np.array([0])

        # allocate array for findopplering
        # init findopplering array to zero
        tree_findoppler = np.zeros(tsteps * tdwidth,dtype=np.float64) + median_flag

        # allocate array for holding original
        # Allocates array in a fast way (without initialize)
        tree_findoppler_original = np.empty_like(tree_findoppler)

        # allocate array for negative doppler rates
        tree_findoppler_flip = np.empty_like(tree_findoppler)

        # build index mask for in-place tree doppler correction
        ibrev = np.zeros(tsteps, dtype=np.int32)

        for i in range(0, tsteps):
            ibrev[i] = bitrev(i, int(np.log2(tsteps)))

##EE: should double check if tdwidth is really better than fftlen here.
        max_val = max_vals()
        if max_val.maxsnr == None:
            max_val.maxsnr = np.zeros(tdwidth, dtype=np.float64)
        if max_val.maxdrift == None:
            max_val.maxdrift = np.zeros(tdwidth, dtype=np.float64)
        if max_val.maxsmooth == None:
            max_val.maxsmooth = np.zeros(tdwidth, dtype='uint8')
        if max_val.maxid == None:
            max_val.maxid = np.zeros(tdwidth, dtype='uint32')
        if max_val.total_n_hits == None:
            max_val.total_n_hits = 0

        #EE: Making "shoulders" to avoid "edge effects". Could do further testing.
        specstart = int(tsteps*shoulder_size/2)
        specend = tdwidth - (tsteps * shoulder_size)

        #--------------------------------
        #Stats calc
        self.the_mean_val, self.the_stddev = comp_stats(spectra.sum(axis=0))

        #--------------------------------
        #Looping over drift_rate_nblock
        #--------------------------------
        drift_rate_nblock = int(np.floor(self.max_drift / (data_obj.drift_rate_resolution*tsteps_valid)))

##EE-debuging        kk = 0

        for drift_block in range(-1*drift_rate_nblock,drift_rate_nblock+1):
            logger.debug( "Drift_block %i"%drift_block)

            #----------------------------------------------------------------------
            # Negative drift rates search.
            #----------------------------------------------------------------------
            if drift_block <= 0:

                #Populates the find_doppler tree with the spectra
                populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,roll=drift_block,reverse=1)

                # populate original array
                np.copyto(tree_findoppler_original, tree_findoppler)

                # populate neg doppler array
                np.copyto(tree_findoppler_flip, tree_findoppler_original)
                
                # Flip matrix across X dimension to search negative doppler drift rates
                FlipX(tree_findoppler_flip, tdwidth, tsteps)
                logger.info("Doppler correcting reverse...")
                tt.taylor_flt(tree_findoppler_flip, tsteps * tdwidth, tsteps)
                logger.debug( "done...")
                
                complete_drift_range = data_obj.drift_rate_resolution*np.array(range(-1*tsteps_valid*(np.abs(drift_block)+1)+1,-1*tsteps_valid*(np.abs(drift_block))+1))
                for k,drift_rate in enumerate(complete_drift_range[(complete_drift_range<self.min_drift) & (complete_drift_range>=-1*self.max_drift)]):
                    # indx  = ibrev[drift_indices[::-1][k]] * tdwidth

                    # DCP 2020.04 -- WAR to drift rate in flipped files
                    if data_obj.header['DELTAF'] < 0:
                        drift_rate *= -1

                    indx  = ibrev[drift_indices[::-1][(complete_drift_range<self.min_drift) & (complete_drift_range>=-1*self.max_drift)][k]] * tdwidth

                    # SEARCH NEGATIVE DRIFT RATES
                    spectrum = tree_findoppler_flip[indx: indx + tdwidth]

                    # normalize
                    spectrum -= self.the_mean_val
                    spectrum /= self.the_stddev

                    #Reverse spectrum back
                    spectrum = spectrum[::-1]

                    n_hits, max_val = hitsearch(spectrum, specstart, specend, self.snr, drift_rate, data_obj.header, fftlen, tdwidth, max_val, 0)
                    info_str = "Found %d hits at drift rate %15.15f\n"%(n_hits, drift_rate)
                    max_val.total_n_hits += n_hits
                    logger.debug(info_str)
                    self.logwriter.info(info_str)

            #----------------------------------------------------------------------
            # Positive drift rates search.
            #----------------------------------------------------------------------
            if drift_block >= 0:

                #Populates the find_doppler tree with the spectra
                populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,
                              roll=drift_block,reverse=1)

                # populate original array
                np.copyto(tree_findoppler_original, tree_findoppler)

                logger.info("Doppler correcting forward...")
                tt.taylor_flt(tree_findoppler, tsteps * tdwidth, tsteps)
                logger.debug( "done...")
                if (tree_findoppler == tree_findoppler_original).all():
                     logger.error("taylor_flt has no effect?")
                else:
                     logger.debug("tree_findoppler changed")

                ##EE: Calculates the range of drift rates for a full drift block.
                complete_drift_range = data_obj.drift_rate_resolution*np.array(range(tsteps_valid*(drift_block),tsteps_valid*(drift_block +1)))

                for k,drift_rate in enumerate(complete_drift_range[(complete_drift_range>=self.min_drift) & (complete_drift_range<=self.max_drift)]):

                    indx  = ibrev[drift_indices[k]] * tdwidth

                    #DCP 2020.04 -- WAR to drift rate in flipped files
                    if data_obj.header['DELTAF'] < 0:
                        drift_rate *= -1

                    # SEARCH POSITIVE DRIFT RATES
                    spectrum = tree_findoppler[indx: indx+tdwidth]

                    # normalize
                    spectrum -= self.the_mean_val
                    spectrum /= self.the_stddev

                    n_hits, max_val = hitsearch(spectrum, specstart, specend, self.snr, drift_rate, data_obj.header, fftlen, tdwidth, max_val, 0)
                    info_str = "Found %d hits at drift rate %15.15f\n"%(n_hits, drift_rate)
                    max_val.total_n_hits += n_hits
                    logger.debug(info_str)
                    self.logwriter.info(info_str)

        # Writing the top hits to file.
        self.filewriter = tophitsearch(tree_findoppler_original, max_val, tsteps, nframes, data_obj.header, tdwidth,
                                       fftlen, self.max_drift,data_obj.obs_length, out_dir = self.out_dir,
                                       logwriter=self.logwriter, filewriter=self.filewriter, obs_info=self.obs_info)

        logger.info("Total number of candidates for coarse channel "+ str(data_obj.header['coarse_chan']) +" is: %i"%max_val.total_n_hits)

#  ======================================================================  #

def populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,roll=0,reverse=0):
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
        np.copyto(tree_findoppler[sind: sind + cplen], np.roll(spectra[i],roll*i*direction))

#         ##EE loads the end part of the current spectrum into the left hand side black region in tree_findoppler (comment below says "next spectra" but for that need i+1...bug?)
         #load end of current spectra into left hand side of next spectra
        sind = i * tdwidth
        np.copyto(tree_findoppler[sind: sind + tsteps*int(shoulder_size/2)], spectra[i, fftlen-(tsteps*int(shoulder_size/2)):fftlen])

    return tree_findoppler



def hitsearch(spectrum, specstart, specend, hitthresh, drift_rate, header, fftlen, tdwidth, max_val, reverse):
    """Searches for hits at given drift rate. A hit occurs in each channel if > hitthresh.

    Args:
      spectrum: ndarray,
      specstart: int,                first index to search for hit in spectrum
      specend: int,                last index to search for hit in spectrum
      hitthresh: float,              signal to noise ratio used as threshold for determining hits
      drift_rate: float,              drift rate at which we are searching for hits
      header: dict,               header in fits header format. See data_handler.py's DATAH5 class header
      fftlen: int,                UNUSED
      tdwidth: int,
      max_val: max_vals,           object to be filled with max values from this search and then returned
      reverse: int(boolean),       used to flag whether fine channel should be reversed

    Returns:
      : int, max_vals,      j is the amount of hits.

    """

    logger.debug('Start searching for hits at drift rate: %f'%drift_rate)
    j = 0
    for i in (spectrum[specstart:specend] > hitthresh).nonzero()[0] + specstart:
        k =  (tdwidth - 1 - i) if reverse else i
        info_str = 'Hit found at SNR %f! %s\t'%(spectrum[i], '(reverse)' if reverse else '')
        info_str += 'Spectrum index: %d, Drift rate: %f\t'%(i, drift_rate)
        info_str += 'Uncorrected frequency: %f\t'%chan_freq(header,  k, tdwidth, 0)
        info_str += 'Corrected frequency: %f'%chan_freq(header, k, tdwidth, 1)
        logger.debug(info_str)
        j += 1
        used_id = j
        if spectrum[i] > max_val.maxsnr[k]:
            max_val.maxsnr[k] = spectrum[i]
            max_val.maxdrift[k] = drift_rate
            max_val.maxid[k] = used_id

    return j, max_val

def tophitsearch(tree_findoppler_original, max_val, tsteps, nframes, header, tdwidth, fftlen,max_drift,obs_length, out_dir='', logwriter=None, filewriter=None,obs_info=None):
    """This finds the hits with largest SNR within 2*tsteps frequency channels.

    Args:
      tree_findoppler_original: ndarray,        spectra-populated findoppler tree
      max_val: max_vals,       contains max values from hitsearch
      tsteps: int,
      nframes: int,            UNUSED
      header: dict,           header in fits header format. See data_handler.py's DATAH5 class header. Used to report tophit in filewriter
      tdwidth: int,
      fftlen: int,            length of fast fourier transform (fft) matrix
      max_drift: float,          Max drift rate in Hz/second
      obs_length: float,
      out_dir: string,         UNUSED (Default value = '')
      logwriter: LogWriter       logwriter to which we should write if we find a top hit (Default value = None)
      filewriter: FileWriter      filewriter corresponding to file to which we should save the
    local maximum of tophit. See report_tophit in filewriters.py (Default value = None)
      obs_info: dict, (Default value = None)

    Returns:
      : FileWriter,     same filewriter that was input

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
                #logwriter.report_tophit(max_val, i, header)
#EE            logger.debug("slice of spectrum...size: %s"%str(tree_orig[0:nframes, lbound:ubound].shape))
            if filewriter:
                filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tdwidth, fftlen, header,max_val.total_n_hits,obs_info=obs_info)
#EE: not passing array cut, since not saving in .dat file                filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tree_orig[0:nframes, lbound:ubound], header)

##EE : Uncomment if want to save each blob              np.save(out_dir + '/spec_drift_%.4f_id_%d.npy'%(max_val.maxdrift[i],i), tree_orig[0:nframes, lbound:ubound])
            else:
                logger.error('Not have filewriter? tell me why.')

    return filewriter
