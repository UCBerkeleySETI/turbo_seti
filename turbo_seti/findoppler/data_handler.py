#!/usr/bin/env python

import os
import numpy as np
import math
from pkg_resources import resource_filename
from pkg_resources import resource_filename
from blimpy import Waterfall
from blimpy.io import sigproc
import h5py

import logging
logger = logging.getLogger(__name__)

#For debugging
#import cProfile
#import pdb;# pdb.set_trace()

SIZE_LIM = 256.0   # File size limit in MB. If larger then make a split mapping.

class DATAHandle:
    """
    """
    def __init__(self, filename=None, size_limit = SIZE_LIM,out_dir='./'):
        self.filename = filename
        if filename and os.path.isfile(filename):
            self.filename = filename
            self.out_dir = out_dir

            if not h5py.is_hdf5(filename):
                if not sigproc.is_filterbank(filename):
                    raise IOError('No correct format, need .h5. Try again...')
                else:
                    logger.info("File .fil detected. Attempting to create .h5 file in current directory...")
                    try:
                        self.__make_h5_file()
                    except:
                        raise IOError('Unable to create .h5 file. Please, try again with correct format.')

            self.filestat = os.stat(filename)
            self.filesize = self.filestat.st_size/(1024.0**2)


            if self.filesize > size_limit:
                logger.info("The file is of size %f MB, exceeding our size limit %f MB. Split needed..."%(self.filesize, size_limit))
                self.data_list = self.__split_h5()
            else:
                #EE This is here mainly for testing. Since need to keep in mind the band pass shape.
                logger.debug("File size %f MB within range %f MB, okay..."%(self.filesize, size_limit))
                data_obj = DATAH5(filename)
                self.data_list = [data_obj]

            self.status = True

        else:
            self.status = False
            logger.error("File %s doesn\'t exists, please check!"%self.filename)
            raise IOError("File %s doesn\'t exists, please check!"%self.filename)

    def get_info(self):
        ''' Returning header.
        '''

        fil_file = Waterfall(self.filename,load_data=False)
        return fil_file.header

    def __make_h5_file(self,):
        ''' Converts file to h5 format. Saves output in current dir.
        '''

        fil_file = Waterfall(self.filename)
        new_filename = self.out_dir+self.filename.replace('.fil','.h5').split('/')[-1]
        fil_file.write_to_hdf5(new_filename)
        self.filename = new_filename

    def __split_h5(self, size_limit=SIZE_LIM):
        '''Creates a plan to select data from single coarse channels.
        '''

        data_list = []

        #Instancing file.
        try:
            fil_file = Waterfall(self.filename)
        except:
            logger.error("Error encountered when trying to open file: %s"%self.filename)
            raise IOError("Error encountered when trying to open file: %s"%self.filename)

        #Finding lowest freq in file.
        f_delt = fil_file.header[b'foff']
        f0 = fil_file.header[b'fch1']

        #Looping over the number of coarse channels.
        n_coarse_chan = int(fil_file.calc_n_coarse_chan())
        if n_coarse_chan != fil_file.calc_n_coarse_chan():
            logger.warning('The file/selection is not an integer number of coarse channels. This could have unexpected consequences. Let op!')

        for chan in range(n_coarse_chan):

            #Calculate freq range for given course channel.
            f_start = f0 + chan*(f_delt)*fil_file.n_channels_in_file/n_coarse_chan
            f_stop = f0 + (chan+1)*(f_delt)*fil_file.n_channels_in_file/n_coarse_chan

            if f_start > f_stop:
                f_start, f_stop = f_stop, f_start

            data_obj = DATAH5(self.filename, f_start=f_start, f_stop=f_stop, coarse_chan=chan, tn_coarse_chan=n_coarse_chan)

#----------------------------------------------------------------

            #This appends to a list of all data instance selections. So that all get processed later.
            data_list.append(data_obj)

        return data_list

class DATAH5:
    ''' This class is where the waterfall data is loaded, as well as the header info.
        It creates other atributes related to the dedoppler search (load_drift_indexes).
    '''

    def __init__(self, filename, size_limit = SIZE_LIM,f_start=None, f_stop=None,t_start=None, t_stop=None,coarse_chan=1,tn_coarse_chan=None):
        self.filename = filename
        self.closed = False
        self.f_start = f_start
        self.f_stop = f_stop
        self.t_start = t_start
        self.t_stop = t_stop
        self.tn_coarse_chan = tn_coarse_chan

        #Instancing file.
        try:
            self.fil_file = Waterfall(filename,f_start=self.f_start, f_stop=self.f_stop,t_start=self.t_start, t_stop=self.t_stop,load_data=False)
        except:
            logger.error("Error encountered when trying to open file %s"%filename)
            raise IOError("Error encountered when trying to open file %s"%filename)

        #Getting header
        try:
            if self.tn_coarse_chan:
                header = self.__make_data_header(self.fil_file.header,coarse=True)
            else:
                header = self.__make_data_header(self.fil_file.header)
        except:
            logger.debug('The fil_file.header is ' % self.fil_file.header)
            raise IOError("Error accessing header from file: %s." % self.filename)

        self.header = header

        self.fftlen = header[b'NAXIS1']
 
        #EE To check if swapping tsteps_valid and tsteps is more appropriate.
        self.tsteps_valid = header[b'NAXIS2']
        self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))
 
        self.obs_length = self.tsteps_valid * header[b'DELTAT']
        self.drift_rate_resolution = (1e6 * np.abs(header[b'DELTAF'])) / self.obs_length   # in Hz/sec
        self.header[b'baryv'] = 0.0
        self.header[b'barya'] = 0.0
        self.header[b'coarse_chan'] = coarse_chan
        
        #EE For now I'm not using a shoulder. This is ok as long as I'm analyzing each coarse channel individually.
        #EE In general this parameter is an integer (even number).
        #This gives two regions, each of n*steps, around spectra[i]
        self.shoulder_size = 0
        self.tdwidth = self.fftlen + self.shoulder_size*self.tsteps

    def load_data(self,):
        ''' Read the data from file.
        '''
        self.fil_file.read_data(f_start=self.f_start, f_stop=self.f_stop)

        #Blanking DC bin.
        n_coarse_chan = int(self.fil_file.calc_n_coarse_chan())
        if n_coarse_chan != self.fil_file.calc_n_coarse_chan():
            logger.warning('The file/selection is not an integer number of coarse channels. This could have unexpected consequences. Let op!')
        self.fil_file.blank_dc(n_coarse_chan)

        spec = np.squeeze(self.fil_file.data)
        spectra = np.array(spec, dtype=np.float64)

        #Arrange data in ascending order in freq if not already in that format.
        if self.header[b'DELTAF'] < 0.0:
            spectra = spectra[:,::-1]

        #This check will add rows of zeros if the obs is too short (and thus not a power of two rows).
        while spectra.shape[0] != self.tsteps:
                spectra = np.append(spectra,np.zeros((1,self.fftlen)),axis=0)
        self.tsteps_valid = self.tsteps
        self.obs_length = self.tsteps * self.header[b'DELTAT']

        if spectra.shape != (self.tsteps_valid, self.fftlen):
            logger.error('Something is wrong with array size.')
            raise IOError('Something is wrong with array size.')

        drift_indexes = self.load_drift_indexes()

        return spectra, drift_indexes

    def load_drift_indexes(self):
        ''' The drift indexes are read from an stored file so that no need to recalculate. This speed things up.
        '''
        n = int(np.log2(self.tsteps))
        di_array = np.genfromtxt(resource_filename('turbo_seti', 'drift_indexes/drift_indexes_array_%d.txt'%n), delimiter=' ', dtype=int)

        ts2 = int(self.tsteps/2)
        drift_indexes = di_array[(self.tsteps_valid - 1 - ts2), 0:self.tsteps_valid]
        return drift_indexes

    def __make_data_header(self,header,coarse=False):
        '''Takes header into fits header format '''

        base_header = {}

        #used by file_writers.py
        base_header[b'SOURCE'] = header[b'source_name'].replace(b'\xc2\xa0',b'_').replace(b' ',b'_')
        base_header[b'MJD'] = header[b'tstart']
        base_header[b'DEC'] = str(header[b'src_dej'])
        base_header[b'RA'] = str(header[b'src_raj'])
        base_header[b'DELTAF'] =  header[b'foff']
        base_header[b'DELTAT'] = float(header[b'tsamp'])

        #used by helper_functions.py
        if coarse:
            base_header[b'NAXIS1'] = int(header[b'nchans']/self.tn_coarse_chan)
            base_header[b'FCNTR'] = (self.f_stop - self.f_start)/2. + self.f_start
        else:
            base_header[b'NAXIS1'] = int(header[b'nchans'])
            base_header[b'FCNTR'] = float(header[b'fch1']) + header[b'foff']*base_header[b'NAXIS1']/2

        #other header values.
        base_header[b'NAXIS'] = 2
        base_header[b'NAXIS2'] = int(self.fil_file.n_ints_in_file)
        return base_header

    def close(self):
        """Closes file and sets the data attribute `.closed` to
        True. A closed object can no longer be used for I/O operations.
        `close()` may be called multiple times without error.
        """

        # Call file object destructor which should close the file
        del self.fil_file

        self.closed = True

