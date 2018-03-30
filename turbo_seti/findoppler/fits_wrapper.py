#!/usr/bin/env python

import astropy.io.fits as pyfits
import os
import glob
import numpy as np
import math
import copy
from pkg_resources import resource_filename
#import barycenter
import onlyreadfilterbank as fr2
import filterbank as fr

import logging
logger = logging.getLogger(__name__)

import pdb;# pdb.set_trace()

class FITSHandle:
    """
    """
    def __init__(self, filename=None, size_limit = 128.0, split_dir='.'):

        if filename and os.path.isfile(filename):
            self.filename = filename
            self.filestat = os.stat(filename)
            filesize = self.filestat.st_size/(1024.0**2)
            if filesize > size_limit:
                logger.info("The file is of size %f MB, exceeding our size limit %f MB. Split needed..."%(filesize, size_limit))

                self.split = True
                if '.fits' in filename:
                    self.fits_list = self.split_fits(filename, split_dir=split_dir, size_limit=size_limit)
                elif '.fil' in filename:
                    self.fits_list = self.split_fil2fits(filename, split_dir=split_dir, size_limit=size_limit)
                else:
                    raise FileError('No .fits or .fil format. Try again...')
            else:
                logger.debug("File size %f MB within range %f MB, okay..."%(filesize, size_limit))
                self.split = False
                fits_obj = nonFITS(filename)  #I'm avoiding spliting the .fil file into fits, this should be the standard way.
                self.fits_list = [fits_obj]
            self.split_filenames = str([fits.filename if fits.status else '' for fits in self.fits_list])
            if self.split_filenames == '[]':
                logger.error("Failed to initiate a FITS instance, aborting...")
                self.status = False
                return None
            else:
                self.status = True
        else:
            raise IOError("File %s doesn\'t exists, please check!"%filename)
#             self.status = False
            return None

    def get_info(self):
        return ""

    @staticmethod
    def split_fits(filename=None, split_dir='.', size_limit = 1024.0):
        fits_list = []
        filestat = os.stat(filename)
        filesize = filestat.st_size/(1024.0**2)
        if filesize <= size_limit:
            logger.error("This file is only %f MB. It is smaller than our size limit %f MB, no split needed."%(filesize, size_limit))
            return []
        try:
            bighdulist = pyfits.open(filename, memmap=True)
            first_row = bighdulist[0]
            header = first_row.header
        except:
            raise IOError("Error encountered when trying to open FITS file %s"%filename)
            return []

        fn = filename[filename.rfind('/')+1:filename.rfind('.fits')]
        deltaf = header['DELTAF']
        fftlen = header['NAXIS1']
        fcntr = header['FCNTR']
        frange = [fcntr - fftlen*deltaf/2, fcntr + fftlen*deltaf/2]

        nfiles_min = int(math.ceil(filesize/size_limit))
        new_width_max = fftlen/nfiles_min
        new_width = 2**math.floor(np.log2(new_width_max))
        nfiles = int(math.ceil(fftlen/new_width))
        new_files = []
        new_fcntrs = []
        new_filenames = []
        indices = []
        new_primary_header = copy.deepcopy(header)
        to_create = []
        for i in range(0, nfiles):
            new_filename = split_dir + '/' + fn + '_%d'%i + '.fits'
            new_filenames.append(new_filename)
            new_fcntr_tmp = frange[0] + deltaf * new_width * (i + 0.5)
            new_fcntrs.append(new_fcntr_tmp)
            new_primary_header['FCNTR'] = new_fcntr_tmp
            ind = (i*new_width, min(fftlen, (i+1)*new_width))
            indices.append(ind)
            if os.path.isfile(new_filename):
                logger.error("file %s already existed!"%new_filename)
                to_create.append(False)
                continue
            to_create.append(True)
            data = first_row.data[0][ind[0]:ind[1]]
            prihdu = pyfits.PrimaryHDU(data, header = new_primary_header)
            prihdu.writeto(new_filename)
            logger.info("Created new file: %s"%new_filename)

        for i, ohdu in enumerate(bighdulist[1:]):
            logger.debug("Dealing with row %d"%i)
            new_header = copy.deepcopy(ohdu.header)
            for j, new_filename in enumerate(new_filenames):
                if not to_create[j]:
                    continue
                new_header['FCNTR'] = new_fcntrs[j]
                ind = indices[j]
                data = ohdu.data[0][ind[0]:ind[1]]
                pyfits.append(new_filename, data, new_header)

##EE This, so that dedoppler is run in all the little new fits files
        for new_filename in new_filenames:
            fits_obj = FITS(new_filename)
            fits_list.append(fits_obj)

        return fits_list

    @staticmethod
    def make_fits_header(header,first=True,LOFAR=False):
        '''Takes .fil header into fits header format '''

        base_header = {}
        base_header['SIMPLE'] = True
        base_header['NAXIS'] = 2
        base_header['NAXIS1'] = int(header['Number of channels'])
        base_header['DOPPLER'] = 0.0
        base_header['SNR'] = 0.0
        base_header['EXTEND'] = True
        base_header['DELTAT'] = float(header['Sample time (us)'])/1e6
        base_header['MJD'] = float(header['Time stamp of first sample (MJD)'])
        base_header['XTENSION'] = 'IMAGE   '
        base_header['PCOUNT'] = 1
        base_header['GCOUNT'] = 1
        base_header['TOFFSET'] = float(header['Sample time (us)'])/1e6

        if LOFAR:
            base_header['BITPIX'] = -32
            base_header['DELTAF'] =  0.000001497456 # LOFAR specific (project LC2_040).
            base_header['DEC'] = float(header['Source DEC (J2000)'])
            base_header['RA'] = float(header['Source RA (J2000)'])
            base_header['SOURCE'] = header['Source Name'].replace('\xc2\xa0','_').replace(' ','_')
        else:
            if  '32' in header['Number of bits per sample']:
                base_header['BITPIX'] = -32
            else:
                raise ValueError('Check nbits per sample. Not equeal 32')
            base_header['DELTAF'] =  np.abs(float(header['Channel bandwidth      (MHz)']))
            base_header['DEC'] = header['Source DEC (J2000)']
            base_header['RA'] = header['Source RA (J2000)']
            base_header['SOURCE'] = header['Source Name'].replace('\xc2\xa0','_').replace(' ','')

        base_header['FCNTR'] = float(header['Frequency of channel 1 (MHz)']) - base_header['DELTAF']*base_header['NAXIS1']/2

        if first:
            base_header['NAXIS2'] = int(header['Number of samples'])
            key_list = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND','DELTAT','DELTAF','FCNTR','MJD','DEC','RA','DOPPLER','SNR','SOURCE']
        else:
            base_header['NAXIS2'] = 1
            key_list = ['XTENSION','BITPIX','NAXIS','NAXIS1','NAXIS2','PCOUNT','GCOUNT','DELTAT','DELTAF','FCNTR','TOFFSET','DEC','RA','DOPPLER','SNR','SOURCE']

        fits_header=pyfits.Header(cards=[pyfits.Card(key=key,value=base_header[key]) for key in key_list])

        return fits_header

    def split_fil2fits(self,filename=None, split_dir='.', size_limit = 128.0):
        ''' Splits (by subbands) LOFAR .fil data into .fits data readable by dedoppler.'''

        fits_list = []
        filestat = os.stat(filename)
        filesize = filestat.st_size/(1024.0**2)

        # Checking if split fits files exist and if their size is bigger or equal to original.
        if glob.glob(filename.replace('.fil','*.fits')):
            size_val=0
            for i in glob.glob(filename.replace('.fil','*.fits')):
                size_val+=os.path.getsize(i)

            if size_val/(1024**2) >= os.path.getsize(filename)/(1024**2):
                logger.info("Files exist and have acceptable size: %s"%filename.replace('.fil','*.fits'))

                #This makes a list of all the new fits files. So that all get processed later.
                for new_filename in glob.glob(filename.replace('.fil','*.fits')):
                    logger.info("Loading fits file: %s"%new_filename)
                    fits_obj = FITS(new_filename)
                    fits_list.append(fits_obj)

                return fits_list

        if filesize <= size_limit:
            logger.error("This file is only %f MB. It is smaller than our size limit %f MB, no split needed."%(filesize, size_limit))
            return []

        try:
            fil_file2=fr2.DataReader(filename)   #tmp, will remove when fully move to filterbank.py
            fil_file=fr.Filterbank(filename)
        except:
            raise IOError("Error encountered when trying to open FIL file %s"%filename)

        try:
#EE_fil            first_row2 = fil_file2.read_row(0)
#            first_row = np.array(first_row,dtype=np.float64)
            first_row = np.array(fil_file.data[0].astype(np.uint32))

            header = self.make_fits_header(fil_file2.headerinfo)
#EE_fil2            header = self.make_fits_header(fil_file.header)
        except:
            logger.error('The fil_file.headerinfo is '%fil_file2.headerinfo)
#EE            logger.error('The header is '%header)
            raise IOError("Error accessing data FIL file %s."%filename)

        fn = filename[filename.rfind('/')+1:filename.rfind('.fil')]
        deltaf = header['DELTAF']
        fftlen = header['NAXIS1']
        fcntr = header['FCNTR']
        frange = [fcntr - fftlen*deltaf/2, fcntr + fftlen*deltaf/2]
        nfiles_min = int(math.ceil(filesize/size_limit))
        new_width_max = fftlen/nfiles_min
        new_width = 2**math.floor(np.log2(new_width_max))
        nfiles = int(math.ceil(fftlen/new_width))
        new_files = []
        new_fcntrs = []
        new_filenames = []
        indices = []
        new_primary_header = copy.deepcopy(header)
        to_create = []


        #Dividing file into finner subbands.
        for i in range(0, nfiles):
            new_filename = split_dir + '/' + fn + '_%d'%i + '.fits'
            new_filenames.append(new_filename)
            new_fcntr_tmp = frange[0] + deltaf * new_width * (i + 0.5)
            new_fcntrs.append(new_fcntr_tmp)
            new_primary_header['FCNTR'] = new_fcntr_tmp
            ind = (i*new_width, min(fftlen, (i+1)*new_width))
            indices.append(ind)
            #Note that this assumes the files exist in complete form (doesn't know about partial files).
            if os.path.isfile(new_filename):
                logger.debug("file %s already existed!"%new_filename)
                to_create.append(False)
                continue
            to_create.append(True)
            data = first_row[0][ind[0]:ind[1]]
            prihdu = pyfits.PrimaryHDU(data, header = new_primary_header)
            prihdu.writeto(new_filename)
            logger.info("Created new file: %s"%new_filename)

        #For speeding up the creation of fits files. First reading a few rows then saving them in each file.
        nrows = 1 # Since (1072-1)/63. = 17.0 and header['NAXIS2'] = 1072 for LC2_040

        ##Only loop over if there is a single file to create, otherwise, skip.
        if np.array(to_create,dtype=bool).any():

            #Looping over blocks of row, filenames and rows per block(nrows).
            for i in range(int(header['NAXIS2']/nrows)-1):
                new_header = self.make_fits_header(fil_file2.headerinfo,first=False)
#EE_fil2                new_header = self.make_fits_header(fil_file.header,first=False)
#EE_fil                next_rows = fil_file.read_rows(i*nrows+1,nrows)
                next_rows = np.array(fil_file.data[i+1].astype(np.uint32))

                for j, new_filename in enumerate(new_filenames):
                    if not to_create[j]:
                        continue

                    hdu_list = pyfits.open(new_filename)

                    for rr in range(nrows):
                        if j == 0:
                            logger.debug("Dealing with row %d"%(i*nrows+1+rr))

                        new_header['TOFFSET'] = (i*nrows+1+rr)*new_header['DELTAT']
                        new_header['FCNTR'] = new_fcntrs[j]
                        ind = indices[j]
                        data = next_rows[rr][int(ind[0]):int(ind[1])]
                        hdu = pyfits.ImageHDU(data, header = new_header)
                        hdu_list.append(hdu)

                    #Overwriting file for maximum speed up (since appending reads the out file first).
                    hdu_list.writeto(new_filename,clobber=True)

        #This makes a list of all the new fits files. So that all get processed later.
        for new_filename in new_filenames:
            fits_obj = FITS(new_filename)
            fits_list.append(fits_obj)

        return fits_list


class  nonFITS:
    """ This class is where the filterbank data is loaded, as well as the header info.
        It creates other atributes related to the search (load_drift_indexes).
        Similar to FITS, but in this case to load fil not fits.

    """
    def __init__(self, filename=None, size_limit = 1024.0):
        if filename and os.path.isfile(filename):
            self.filename = filename
            self.filestat = os.stat(filename)
            filesize = self.filestat.st_size/(1024.0**2)
            if filesize > size_limit:
                logger.error("The file is of size %f MB, exceeding our size limit %f MB. Aborting..."%(filesize, size_limit))
                return None
            try:
                fil_file2=fr2.DataReader(filename)  # Will be replaced by danny's filterbank...
                fil_file=fr.Filterbank(filename)
                header = self.make_fits_header(fil_file2.headerinfo)
#EE_fil2                header = self.make_fits_header(fil_file.header)
            except:
                logger.error("Error encountered when trying to open FITS file %s"%filename)
                self.status = False
                return None
            self.fftlen = header['NAXIS1']
            self.tsteps_valid = header['NAXIS2']
            self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))   ## what is this for??
            self.obs_length = self.tsteps_valid * header['DELTAT']
            self.shoulder_size = 0
            self.tdwidth = self.fftlen + self.shoulder_size*self.tsteps  ##EE why is this multiplied by 8? This gives two regions, each of 4*steps, around spectra[i]
            self.drift_rate_resolution = (1e6 * header['DELTAF']) / self.obs_length
            self.nom_max_drift = self.drift_rate_resolution * self.tsteps_valid  ##EE Do I need tsteps here?

            ##EE: debug. Skyping barycenter for now. Would need to debug it first.
            if logger.getEffectiveLevel() > 1000:  ##EE: logging.getLevelName(10)='DEBUG'

                self.header = barycenter.correct(header, self.obs_length)
                logger.info('barycenter done for fits file %s! baryv: %f'%(filename, self.header['baryv']))
            else:
                self.header = header
                self.header['baryv'] = 0.0
                self.header['barya'] = 0.0

            # some default values
            self.original_vals= {'tsteps_valid': self.tsteps_valid, 'tsteps': self.tsteps,
                                 'tdwidth': self.tdwidth, 'fftlen':self.fftlen}
            self.compressed_t = False
            self.compressed_f = False
            self.status = True

    @staticmethod
    def make_fits_header(header,LOFAR=False):
        '''Takes .fil header into fits header format '''

        base_header = {}
        base_header['SIMPLE'] = True
        base_header['NAXIS'] = 2

        base_header['DOPPLER'] = 0.0
        base_header['SNR'] = 0.0
        base_header['EXTEND'] = True
        base_header['XTENSION'] = 'IMAGE   '
        base_header['PCOUNT'] = 1
        base_header['GCOUNT'] = 1

        if '32' in header['Number of bits per sample']:
            base_header['BITPIX'] = -32
        else:
            raise ValueError('Check nbits per sample. Not equeal 32')

        base_header['NAXIS1'] = int(header['Number of channels'])  #nchans
#EE_fil2        base_header['NAXIS1'] = int(header['nchans'])  #nchans
        base_header['NAXIS2'] = int(header['Number of samples'])
#EE_fil2        base_header['NAXIS2'] = int(header[''])
        base_header['DELTAT'] = float(header['Sample time (us)'])/1e6
        base_header['MJD'] = float(header['Time stamp of first sample (MJD)'])
        base_header['TOFFSET'] = float(header['Sample time (us)'])/1e6
#EE_fil2        base_header['DELTAT'] = float(header['tsamp'])
#EE_fil2        base_header['MJD'] = float(header['tstart'])
#EE_fil2        base_header['TOFFSET'] = float(header['tsamp)'])
        base_header['DELTAF'] =  np.abs(float(header['Channel bandwidth      (MHz)']))
#EE_fil2        base_header['DELTAF'] =  np.abs(float(header['foff']))
        base_header['SOURCE'] = header['Source Name'].replace('\xc2\xa0','_').replace(' ','')  #Removing white spaces and bad formats
#EE_fil2        base_header['SOURCE'] = header['source_name'].replace('\xc2\xa0','_').replace(' ','')   #Removing white spaces and bad formats
        base_header['FCNTR'] = float(header['Frequency of channel 1 (MHz)']) - base_header['DELTAF']*base_header['NAXIS1']/2
#EE_fil2        base_header['FCNTR'] = float(header['fch1']) - base_header['DELTAF']*base_header['NAXIS1']/2
        base_header['DEC'] = float(header['Source DEC (J2000)'])
        base_header['RA'] = float(header['Source RA (J2000)'])
#EE_fil2        base_header['DEC'] = float(header['Source DEC (J2000)'])
#EE_fil2        base_header['RA'] = float(header['Source RA (J2000)'])

        return base_header

    def load_data(self, max_search_rate=None, bw_compress_width=None, logwriter=None):
        ''' Read all the data from file.
        '''

#EE_fil        fil_file = fr2.DataReader(self.filename)
        fil_file = fr.Filterbank(filename)
#EE_fil        spec = fil_file.read_all()
        spec = np.squeeze(fil_file.data)
        spectra = np.array(spec, dtype=np.float64)

        if spectra.shape != (self.tsteps_valid, self.fftlen):
            raise ValueError('Something is wrong with array size.')

        drift_indexes = self.load_drift_indexes()

        return spectra, drift_indexes

    def load_drift_indexes(self):
        ''' The drift indexes are read from an stored file so that no need to recalculate. This speed things up.
        '''
        n = int(np.log2(self.tsteps))
        if n > 9:
            di_array = np.genfromtxt(resource_filename('dedoppler_bones', '../drift_indexes/drift_indexes_array_%d.txt'%n), delimiter=' ', dtype=int)
        else:
            di_array = np.genfromtxt(resource_filename('dedoppler_bones', '../drift_indexes/drift_indexes_array_%d.txt'%n), delimiter='\t', dtype=int)

        ts2 = self.tsteps/2
        drift_indexes = di_array[self.tsteps_valid - 1 - ts2, 0:self.tsteps_valid]
        return drift_indexes

    def get_info(self):
        return ""


class FITS:
    """ This class is where the filterbank data is loaded, as well as the header info.
        It creates other atributes related to the search (load_drift_indexes).

    """
    def __init__(self, filename=None, size_limit = 1024.0):
        if filename and os.path.isfile(filename):
            self.filename = filename
            self.filestat = os.stat(filename)
            filesize = self.filestat.st_size/(1024.0**2)
            if filesize > size_limit:
                logger.error("The file is of size %f MB, exceeding our size limit %f MB. Aborting..."%(filesize, size_limit))
                return None
            try:
                hdulist = pyfits.open(filename, memmap=True)
                #self.info = hdulist.info()
                first_row = hdulist[0]
                header = first_row.header
                hdulist.close()
            except:
                logger.error("Error encountered when trying to open FITS file %s"%filename)
                self.status = False
                return None
            self.fftlen = header['NAXIS1']
            self.tsteps_valid = len(hdulist)
            self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))   ## what is this for??
            self.obs_length = self.tsteps_valid * header['DELTAT']
            self.shoulder_size = 0
            self.tdwidth = self.fftlen + self.shoulder_size*self.tsteps  ##EE why is this multiplied by 8? This gives two regions, each of 4*steps, around spectra[i]
            self.drift_rate_resolution = (1e6 * header['DELTAF']) / self.obs_length
            self.nom_max_drift = self.drift_rate_resolution * self.tsteps_valid  ##EE Do I need tsteps here?

##EE: debug. Skyping barycenter for now. Would need to debug it first.
            if logger.getEffectiveLevel() > 1000:  ##EE: logging.getLevelName(10)='DEBUG'

                self.header = barycenter.correct(header, self.obs_length)
                logger.info('barycenter done for fits file %s! baryv: %f'%(filename, self.header['baryv']))
            else:
                    self.header = header
                    self.header['baryv'] = 0.0
                    self.header['barya'] = 0.0

            # some default values
            self.original_vals= {'tsteps_valid': self.tsteps_valid, 'tsteps': self.tsteps,
                                 'tdwidth': self.tdwidth, 'fftlen':self.fftlen}
            self.compressed_t = False
            self.compressed_f = False
            self.status = True

    def load_data(self, max_search_rate=None, bw_compress_width=None, logwriter=None):
##        hdulist = pyfits.open(self.filename, memmap=True)
        hdulist = pyfits.open(self.filename)
        spectra = np.empty((self.tsteps_valid, self.fftlen), dtype=np.float64)

        if max_search_rate and self.nom_max_drift > max_search_rate:
            ##EE I'm stoping here until I know what this do.
            STOP

            logger.info("nominal max drift rate greater than allowed.... decimating.")
            self.compressed_t = True
            decimate_factor = math.floor(self.nom_max_drift / max_search_rate)
            self.tsteps_valid = int((self.tsteps_valid - decimate_factor)/decimate_factor)
            self.tsteps = int(math.pow(2, math.ceil(np.log2(math.floor(self.tsteps_valid)))))
            self.tdwidth = self.fftlen + 8*self.tsteps
            self.nom_max_drift = self.drift_rate_resolution * self.tsteps_valid
            for i in range(0, self.tsteps_valid+1):
#                print "loading row %d..."%(i*decimate_factor)
                np.copyto(spectra[i], hdulist[i*decimate_factor].data[0])
                for k in range(1, decimate_factor):
                    # ???
                    logger.debug("Decimation: adding row %d"%(i*decimate_factor + k))
                    spectra[i, :] += hdulist[i*decimate_factor + k].data[0]
                spectra = spectra[0:self.tsteps_valid, :]
        else:
            for i in range(0, self.tsteps_valid):
                logger.debug("loading row %d"%i)
                np.copyto(spectra[i], np.squeeze(hdulist[i].data))
                ##EE np.squeeze is used since the shape of FITS before or after spliting is not the same.
                ##EE This takes care of the issue when dimensions are (1,n)

        compressed_spectra = None
        if bw_compress_width:
            compressed_spectra = matrix_compression(spectra, compression_width = bw_compress_width, axis=1, method='max')
        if compressed_spectra is None:
            compressed_spectra = spectra
        else:
            self.compressed_f = True
            self.fftlen = compressed_spectra.shape[-1]
            self.tdwidth = self.fftlen + 8*self.tsteps

        drift_indexes = self.load_drift_indexes()
        return compressed_spectra, drift_indexes

    def load_drift_indexes(self):
        ''' The drift indexes are read from an stored file so that no need to recalculate. This speed things up.
        '''
        n = int(np.log2(self.tsteps))
        if n > 9:
            di_array = np.genfromtxt(resource_filename('dedoppler_bones', '../drift_indexes/drift_indexes_array_%d.txt'%n), delimiter=' ', dtype=int)
        else:
            di_array = np.genfromtxt(resource_filename('dedoppler_bones', '../drift_indexes/drift_indexes_array_%d.txt'%n), delimiter='\t', dtype=int)

        ts2 = self.tsteps/2
        drift_indexes = di_array[self.tsteps_valid - 1 - ts2, 0:self.tsteps_valid]
        return drift_indexes

    def get_info(self):
        return ""


def matrix_compression(matrix_original, compression_width=1, axis=0, method='add'):
    """
    matrix_original: assumed to be a well-formed 2-d array
    compression_width: an integer
    axis: 0 or 1, if 0 sum along column, if 1 then sum along row
    """
    matrix_shape = matrix_original.shape
    if len(matrix_shape) != 2:
        logger.error("Sorry, this function can only handle 2-dim arrays, aborting...")
        return None
    compression_width = int(max(1, math.floor(compression_width)))
    if compression_width > 1 and compression_width < matrix_shape[axis]:
        target_ind = range(0,  int(matrix_shape[axis]), compression_width)
        if method=='max':
            matrix_compressed = np.empty_like(matrix_original[:, target_ind] if axis else matrix_original[target_ind, :])
            for i, val in enumerate(target_ind):
                temp = np.amax(matrix_original[:, val:val+compression_width] if axis else matrix_original[val:val+compression_width], axis=axis)
                np.copyto( matrix_compressed[:, i] if axis else matrix_compressed[i, :], temp)
        elif method=='add':
            matrix_compressed = matrix_original[:, target_ind] if axis else  matrix_original[target_ind, :]
            for i in target_ind:
                ind1 = int(i/compression_width)
                if axis: # sum along rows
                    np.copyto(matrix_compressed[:, ind1], matrix_original[:, i])
                else:
                    np.copyto(matrix_compressed[ind1, :], matrix_original[i, :])
                for j in range(1, compression_width):
                    ind2 = i + j
                    if (not axis) and ind2 < matrix_shape[axis]:
                        matrix_compressed[ind1, :] += matrix_original[ind2, :]
                    elif ind2 < matrix_shape[axis]:
                        matrix_compressed[:, ind1] += matrix_original[:, ind2]
        else:
            logger.error("Method unkonwn/unimplmeneted, returning None...")
            return
        logger.info("Compression done, width: %d, axis: %d"%(compression_width, axis))
    else:
        logger.info("No compression to be performed, returning None...")
        matrix_compressed = None
    return matrix_compressed
