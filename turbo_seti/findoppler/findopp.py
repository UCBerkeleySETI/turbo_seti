#!/usr/bin/env python


import numpy as np
import sys
import os
import logging
logger = logging.getLogger(__name__)
import gc   #Garbage collector.


try:
    from .data_handler import DATAHandle
    from .file_writers import FileWriter, LogWriter
    from .helper_functions import chan_freq
except:
    from data_handler import DATAHandle
    from file_writers import FileWriter, LogWriter
    from helper_functions import chan_freq

#For importing cython code
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

try:
    from . import taylor_tree as tt
except:
    import taylor_tree as tt

#For debugging
#import cProfile
#import pdb;# pdb.set_trace()

class max_vals:
    def __init__(self):
        self.maxsnr = None
        self.maxdrift = None
        self.maxsmooth = None
        self.maxid = None
        self.total_n_candi = None

class hist_vals:
    ''' Temporary class that saved the normalized spectrum for all drift rates.'''
    def __init__(self):
        self.histsnr = None
        self.histdrift = None
        self.histid = None

class FinDoppler:
    """ """
    def __init__(self, datafile, max_drift, min_drift = 0, snr = 25.0, out_dir = './', coarse_chans = None, obs_info = None, flagging = None):
        self.min_drift = min_drift
        self.max_drift = max_drift
        self.snr = snr
        self.out_dir = out_dir
        self.data_handle = DATAHandle(datafile,out_dir=out_dir)
        if (self.data_handle is None) or (self.data_handle.status is False):
            raise IOError("File error, aborting...")
        if coarse_chans:
            if int(coarse_chans[-1]) > len(self.data_handle.data_list) or int(coarse_chans[0]) > len(self.data_handle.data_list):
                raise ValueError('The coarse channel(s) given (%i,%i) is outside the possible range (0,%i) '%(int(coarse_chans[0]),int(coarse_chans[-1]),len(self.data_handle.data_list)))
            if int(coarse_chans[-1]) < 0 or int(coarse_chans[0]) < 0:
                raise ValueError('The coarse channel(s) given (%i,%i) is outside the possible range (0,%i) '%(int(coarse_chans[0]),int(coarse_chans[-1]),len(self.data_handle.data_list)))
            self.data_handle.data_list = self.data_handle.data_list[int(coarse_chans[0]):int(coarse_chans[-1])]
        logger.info(self.data_handle.get_info())
        logger.info("A new FinDoppler instance created!")
        self.obs_info = obs_info
        self.status = True
        self.flagging = flagging

    def get_info(self):
        info_str = "File: %s\n drift rates (min, max): (%f, %f)\n SNR: %f\n"%(self.data_handle.filename, self.min_drift, self.max_drift,self.snr)
        return info_str

    def search(self):
        '''Top level search.
        '''
        logger.debug("Start searching...")
        logger.debug(self.get_info())

        self.logwriter = LogWriter('%s/%s.log'%(self.out_dir.rstrip('/'), self.data_handle.data_list[0].filename.split('/')[-1].replace('.h5','').replace('.fits','').replace('.fil','')))
        self.filewriter = FileWriter('%s/%s.dat'%(self.out_dir.rstrip('/'), self.data_handle.data_list[0].filename.split('/')[-1].replace('.h5','').replace('.fits','').replace('.fil','')),self.data_handle.data_list[0].header)

        logger.info("Start ET search for %s"%self.data_handle.data_list[0].filename)
        self.logwriter.info("Start ET search for %s"%(self.data_handle.data_list[0].filename))

        for ii,target_data_obj in enumerate(self.data_handle.data_list):
            self.search_data(target_data_obj)
##EE-benshmark            cProfile.runctx('self.search_data(target_data_obj)',globals(),locals(),filename='profile_M%2.1f_S%2.1f_t%i'%(self.max_drift,self.snr,int(os.times()[-1])))

            #----------------------------------------
            #Closing instance. Collect garbage.
            self.data_handle.data_list[ii].close()
            gc.collect()

    def search_data(self, data_obj):
        '''
        '''

        logger.info("Start searching for coarse channel: %s"%data_obj.header[u'coarse_chan'])
        self.logwriter.info("Start searching for %s ; coarse channel: %i "%(data_obj.filename,data_obj.header[u'coarse_chan']))
        spectra, drift_indexes = data_obj.load_data()
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

        #/* allocate array for negative doppler rates */
        tree_findoppler_flip = np.empty_like(tree_findoppler)

        #/* build index mask for in-place tree doppler correction */
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
        if max_val.total_n_candi == None:
            max_val.total_n_candi = 0

##EE-debuging
#         hist_val = hist_vals()
#         hist_len = int(np.ceil(2*(self.max_drift-self.min_drift)/data_obj.drift_rate_resolution))
#         if hist_val.histsnr == None:
#             hist_val.histsnr = np.zeros((hist_len,tdwidth), dtype=np.float64)
#         if hist_val.histdrift == None:
#             hist_val.histdrift = np.zeros((hist_len), dtype=np.float64)
#         if hist_val.histid == None:
#             hist_val.histid = np.zeros(tdwidth, dtype='uint32')

        #EE: Making "shoulders" to avoid "edge effects". Could do further testing.
        specstart = (tsteps*shoulder_size/2)
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

                #Populates the findoppler tree with the spectra
                populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,roll=drift_block,reverse=1)

                #/* populate original array */
                np.copyto(tree_findoppler_original, tree_findoppler)

                #/* populate neg doppler array */
                np.copyto(tree_findoppler_flip, tree_findoppler_original)

                #/* Flip matrix across X dimension to search negative doppler drift rates */
                FlipX(tree_findoppler_flip, tdwidth, tsteps)

                logger.info("Doppler correcting reverse...")
                tt.taylor_flt(tree_findoppler_flip, tsteps * tdwidth, tsteps)
                logger.debug( "done...")

                complete_drift_range = data_obj.drift_rate_resolution*np.array(range(-1*tsteps_valid*(np.abs(drift_block)+1)+1,-1*tsteps_valid*(np.abs(drift_block))+1))

                for k,drift_rate in enumerate(complete_drift_range[(complete_drift_range<self.min_drift) & (complete_drift_range>=-1*self.max_drift)]):
                    indx  = ibrev[drift_indexes[::-1][(complete_drift_range<self.min_drift) & (complete_drift_range>=-1*self.max_drift)][k]] * tdwidth

                    #/* SEARCH NEGATIVE DRIFT RATES */
                    spectrum = tree_findoppler_flip[indx: indx + tdwidth]

                    #/* normalize */
                    spectrum -= self.the_mean_val
                    spectrum /= self.the_stddev

                    #Reverse spectrum back
                    spectrum = spectrum[::-1]

##EE old wrong use of reverse            n_candi, max_val = candsearch(spectrum, specstart, specend, self.snr, drift_rate, data_obj.header, fftlen, tdwidth, channel, max_val, 1)
                    n_candi, max_val = candsearch(spectrum, specstart, specend, self.snr, drift_rate, data_obj.header, fftlen, tdwidth, max_val, 0)
                    info_str = "Found %d candidates at drift rate %15.15f\n"%(n_candi, drift_rate)
                    max_val.total_n_candi += n_candi
                    logger.debug(info_str)
                    self.logwriter.info(info_str)

##EE-debuging                    np.save(self.out_dir + '/spectrum_dr%f.npy'%(drift_rate),spectrum)

##EE-debuging                    hist_val.histsnr[kk] = spectrum
##EE-debuging                    hist_val.histdrift[kk] = drift_rate
##EE-debuging                    kk+=1

            #----------------------------------------------------------------------
            # Positive drift rates search.
            #----------------------------------------------------------------------
            if drift_block >= 0:

                #Populates the findoppler tree with the spectra
                populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,roll=drift_block,reverse=1)

                #/* populate original array */
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

                    indx  = (ibrev[drift_indexes[k]] * tdwidth)
                    #/* SEARCH POSITIVE DRIFT RATES */
                    spectrum = tree_findoppler[indx: indx+tdwidth]

                    #/* normalize */
                    spectrum -= self.the_mean_val
                    spectrum /= self.the_stddev

                    n_candi, max_val = candsearch(spectrum, specstart, specend, self.snr, drift_rate, data_obj.header, fftlen, tdwidth, max_val, 0)
                    info_str = "Found %d candidates at drift rate %15.15f\n"%(n_candi, drift_rate)
                    max_val.total_n_candi += n_candi
                    logger.debug(info_str)
                    self.logwriter.info(info_str)

                    #-------

##EE-debuging                    np.save(self.out_dir + '/spectrum_dr%f.npy'%(drift_rate),spectrum)

##EE-debuging                    hist_val.histsnr[kk] = spectrum
##EE-debuging                    hist_val.histdrift[kk] = drift_rate
##EE-debuging                    kk+=1
        #-------
##EE-debuging        np.save(self.out_dir + '/histsnr.npy', hist_val.histsnr)
##EE-debuging        np.save(self.out_dir + '/histdrift.npy', hist_val.histdrift)

        #----------------------------------------
        # Writing the top hits to file.

#         self.filewriter.report_coarse_channel(data_obj.header,max_val.total_n_candi)
        self.filewriter = tophitsearch(tree_findoppler_original, max_val, tsteps, nframes, data_obj.header, tdwidth, fftlen, self.max_drift,data_obj.obs_length, out_dir = self.out_dir, logwriter=self.logwriter, filewriter=self.filewriter, obs_info = self.obs_info)
        logger.info("Total number of candidates for coarse channel "+ str(data_obj.header[u'coarse_chan']) +" is: %i"%max_val.total_n_candi)


#  ======================================================================  #

def populate_tree(spectra,tree_findoppler,nframes,tdwidth,tsteps,fftlen,shoulder_size,roll=0,reverse=0):
    """ This script populates the findoppler tree with the spectra.
        It creates two "shoulders" (each region of tsteps*(shoulder_size/2) in size) to avoid "edge" issues.
        It uses np.roll() for drift-rate blocks higher than 1.
    """

    if reverse:
        direction = -1
    else:
        direction = 1

#EE Also, the shouldering is maybe not important, since I'm already making my own flagging fo 10k channels
#EE And Since , I have a very large frequency number comparted to the time lenght.
##EE Wondering if here should have a data cube instead...maybe not, i guess this is related to the bit-reversal.

    for i in range(0, nframes):
        sind = tdwidth*i + tsteps*shoulder_size/2
        cplen = fftlen

        ##EE copy spectra into tree_findoppler, leaving two regions in each side blanck (each region of tsteps*(shoulder_size/2) in size).
#        np.copyto(tree_findoppler[sind: sind + cplen], spectra[i])
        # Copy spectra into tree_findoppler, with rolling.
        np.copyto(tree_findoppler[sind: sind + cplen], np.roll(spectra[i],roll*i*direction))

#EE code below will be replaced, since I think is better to use the median of the data as "flag" than to add other data into the shoulders.

#         ##EE loads the end part of the current spectrum into the left hand side black region in tree_findoppler (comment below says "next spectra" but for that need i+1...bug?)
         #//load end of current spectra into left hand side of next spectra
        sind = i * tdwidth
        np.copyto(tree_findoppler[sind: sind + tsteps*shoulder_size/2], spectra[i, fftlen-(tsteps*shoulder_size/2):fftlen])

    return tree_findoppler

#  ======================================================================  #
#  This function bit-reverses the given value "inval" with the number of   #
#  bits, "nbits".    ----  R. Ramachandran, 10-Nov-97, nfra.               #
#  python version ----  H. Chen   Modified 2014                            #
#  ======================================================================  #
def bitrev(inval, nbits):
    if nbits <= 1:
        ibitr = inval
    else:
        ifact = 1
        for i in range(1, nbits):
           ifact *= 2
        k = inval
        ibitr = (1 & k) * ifact
        for i in range(2, nbits+1):
            k /= 2
            ifact /= 2
            ibitr += (1 & k) * ifact
    return ibitr

#  ======================================================================  #
#  This function bit-reverses the given value "inval" with the number of   #
#  bits, "nbits".                                                          #
#  python version ----  H. Chen   Modified 2014                            #
#  reference: stackoverflow.com/questions/12681945                         #
#  ======================================================================  #
def bitrev2(inval, nbits, width=32):
    b = '{:0{width}b}'.format(inval, width=width)
    ibitr = int(b[-1:(width-1-nbits):-1], 2)
    return ibitr

#  ======================================================================  #
#  This function bit-reverses the given value "inval" with 32bits    #
#  python version ----  E.Enriquez   Modified 2016                            #
#  reference: stackoverflow.com/questions/12681945                         #
#  ======================================================================  #
def bitrev3(x):
    raise DeprecationWarning("WARNING: This needs testing.")

    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16)
    return x

def AxisSwap(inbuf, outbuf, nchans, NTSampInRead):
    #long int    j1, j2, indx, jndx;
    for j1 in range(0, NTSampInRead):
        indx  = j1 * nchans
        for j2 in range(nchans-1, -1, -1):
            jndx = j2 * NTSampInRead + j1
            outbuf[jndx]  = inbuf[indx+j2]

def FlipBand(outbuf, nchans, NTSampInRead):
    temp = np.zeros(nchans*NTSampInRead, dtype=np.float64)

    indx  = (nchans - 1);
    for i in range(0, nchans):
        jndx = (indx - i) * NTSampInRead
        kndx = i * NTSampInRead
        np.copyto(temp[jndx: jndx+NTSampInRead], outbuf[kndx + NTSampInRead])
    #memcpy(outbuf, temp, (sizeof(float)*NTSampInRead * nchans));
    outbuf = temp
    return

def FlipX(outbuf, xdim, ydim):
    temp = np.empty_like(outbuf[0:xdim])
    logger.debug("FlipX: temp array dimension: %s"%str(temp.shape))

    for j in range(0, ydim):
        indx = j * xdim
        np.copyto(temp, outbuf[indx:indx+xdim])
        np.copyto(outbuf[indx: indx+xdim], temp[::-1])
    return

def comp_stats(arrey):
    #Compute mean and stddev of floating point vector array in a fast way, without using the outliers.

    new_vec = np.sort(arrey,axis=None)

    #Removing the lowest 5% and highest 5% of data, this takes care of outliers.
    new_vec = new_vec[int(len(new_vec)*.05):int(len(new_vec)*.95)]
    the_median = np.median(new_vec)
    the_stddev = new_vec.std()

    return the_median, the_stddev

def candsearch(spectrum, specstart, specend, candthresh, drift_rate, header, fftlen, tdwidth, max_val, reverse):
    ''' Searches for candidates: each channel if > candthresh.
    '''

    logger.debug('Start searching for drift rate: %f'%drift_rate)
    j = 0
    for i in (spectrum[specstart:specend] > candthresh).nonzero()[0] + specstart:
        k =  (tdwidth - 1 - i) if reverse else i
        info_str = 'Candidate found at SNR %f! %s\t'%(spectrum[i], '(reverse)' if reverse else '')
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
    '''This finds the candidate with largest SNR within 2*tsteps frequency channels.
    '''

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
                filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tdwidth, fftlen, header,max_val.total_n_candi,obs_info=obs_info)
#EE: not passing array cut, since not saving in .dat file                filewriter = filewriter.report_tophit(max_val, i, (lbound, ubound), tree_orig[0:nframes, lbound:ubound], header)

##EE : Uncomment if want to save each blob              np.save(out_dir + '/spec_drift_%.4f_id_%d.npy'%(max_val.maxdrift[i],i), tree_orig[0:nframes, lbound:ubound])
            else:
                logger.error('Not have filewriter? tell me why.')

    return filewriter
