'''
Download a set of HDF5 files (H5_FILE_LIST)
from the specified web address (URL_DIR)
to the file system directory indicated by DATADIR.
'''

from time import time
import sys
from shutil import rmtree
from os import mkdir
from urllib.error import HTTPError


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


try:
    import wget
except:
    oops('Needed: python3 -m pip install wget')


DATADIR = '/tmp/turbo_seti_data/' # <--------- H5, DAT, LOG directory

URL_DIR = 'http://blpd0.ssl.berkeley.edu/parkes_testing/'

H5_FILE_LIST = ['diced_Parkes_57941_12846_HIP33499_S_fine.h5',
                'diced_Parkes_57941_13194_HIP33499_R_fine.h5',
                'diced_Parkes_57941_13542_HIP33499_S_fine.h5',
                'diced_Parkes_57941_13884_HIP33499_R_fine.h5',
                'diced_Parkes_57941_14233_HIP33499_S_fine.h5',
                'diced_Parkes_57941_14584_HIP33499_R_fine.h5']


def wgetter(arg_h5_name):
    '''
    wget an HDF5 file from the Internet repository.
    arg_h5_name:  HDF5 file name
    '''
    url_h5 = URL_DIR + arg_h5_name
    path_h5 = DATADIR + arg_h5_name
    print('wgetter: Begin wget {} -> {} .....'.format(url_h5, path_h5))
    time_start = time()
    try:
        wget.download(url_h5, path_h5, bar=False)
    except HTTPError as ex:
        oops('wgetter: wget {}, failed: {}'.format(url_h5, repr(ex)))
    time_stop = time()
    print('wgetter: End wget ({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))


rmtree(DATADIR, ignore_errors=True)
mkdir(DATADIR)

for filename_h5 in H5_FILE_LIST:
    wgetter(filename_h5)
