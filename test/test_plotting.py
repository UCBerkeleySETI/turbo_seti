'''
Package turbo_seti
test_plotting.py - Expand coverage for individual files in the
                   turbo_seti/find_event folder, especially plotting.

Date         Who            Description
----------   -------------  --------------------------------------------
2020-07-30   R. Elkins      Initial version
'''

from time import time
from shutil import rmtree
from os import mkdir
from tempfile import gettempdir
import sys
from urllib.error import HTTPError
import wget

from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline


FLAG_INIT = True
FLAG_WGETTER = True
FLAG_MAKE_ALL_DAT_FILES = True
FLAG_PROGRESS_BAR = wget.bar_adaptive   # for a progress bar
FLAG_PROGRESS_BAR = False               # for a quiet run


URL_DIR = 'http://blpd0.ssl.berkeley.edu/parkes_testing/'
FILE_LIST = ['diced_Parkes_57941_12846_HIP33499_S_fine.h5',
             'diced_Parkes_57941_13194_HIP33499_R_fine.h5',
             'diced_Parkes_57941_13542_HIP33499_S_fine.h5',
             'diced_Parkes_57941_13884_HIP33499_R_fine.h5',
             'diced_Parkes_57941_14233_HIP33499_S_fine.h5',
             'diced_Parkes_57941_14584_HIP33499_R_fine.h5']
TESTDIR = gettempdir() + '/turbo_seti_testing/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files.lst'
PATH_CSVF = TESTDIR + 'found_event_table.csv'


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


def initialize():
    '''
    Initialize:
    * Recreate working directory, TESTDIR.
    * Download the H5 file to it.
    '''
    rmtree(TESTDIR, ignore_errors=True)
    mkdir(TESTDIR)
    # Somehow, some way, "testing" is inserted between dirpath
    # and the actual file name.  How?????
    mkdir(TESTDIR + '/testing')
    print('test_plotting: initialized')

def wgetter(arg_h5_name, arg_bar=False):
    '''
    wget an HDF5 file from the Internet repository.
    arg_h5_name:  HDF5 file name
    '''
    print('test_plotting: wget begin ({}) ...'.format(arg_h5_name))
    url_h5 = URL_DIR + arg_h5_name
    path_h5 = TESTDIR + arg_h5_name
    print('test_plotting: Begin wget {} -> {} .....'.format(url_h5, path_h5))
    time_start = time()
    try:
        wget.download(url_h5, path_h5, bar=arg_bar)
    except HTTPError as ex:
        oops('test_plotting: wget {}, failed: {}'.format(url_h5, repr(ex)))
    time_stop = time()
    print('test_plotting: wget ({}) complete, et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))


def make_one_dat_file(arg_h5_name):
    '''
    Make a single DAT file:
    * Instantiate the FindDoppler class object.
    * With the object, search the H5, creating the DAT file
      and a LOG file (not used).

    Note that a max drift of 1 assumes a drift rate of +/- 1
    SNR threshold = 25
    '''
    print('test_plotting: Begin FindDoppler({}) .....'.format(arg_h5_name))
    h5_path = TESTDIR + arg_h5_name
    time_start = time()
    doppler = FindDoppler(h5_path,
                          max_drift=1,
                          min_drift=-1,
                          snr=25,
                          out_dir=TESTDIR)
    time_stop = time()
    print('test_plotting: End FindDoppler({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))

    print('test_plotting: Begin Doppler search({}) .....'
          .format(arg_h5_name))
    time_start = time()
    doppler.search()
    time_stop = time()
    print('test_plotting: End Doppler search({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))


def make_all_dat_files():
    '''
    For each HDF5 file name,
    * Make one DAT file.
    * Add its name to the list of DAT files.
    '''
    with open(PATH_DAT_LIST_FILE, 'w') as file_handle:
        for filename_h5 in FILE_LIST:
            make_one_dat_file(filename_h5)
            filename_dat = filename_h5.replace('.h5', '.dat')
            file_handle.write('{}\n'.format(TESTDIR + filename_dat))


def test_plotting(arg_bar=False):
    '''
    Main procedure
    '''

    main_time_start = time()

    # If configured to do so, initialize temp directory.
    if FLAG_INIT:
        initialize()

    # If configured to do so, fetch all of the HDF5 files from the Internet.
    if FLAG_WGETTER:
        for filename_h5 in FILE_LIST:
            wgetter(filename_h5, arg_bar)

    # If configured to do so, make all of the DAT files.
    if FLAG_MAKE_ALL_DAT_FILES:
        make_all_dat_files()

    # Create the CSV file used by plot_event_pipeline.
    print('test_plotting: find_event_pipeline({}) ...'
          .format(PATH_DAT_LIST_FILE))
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   filter_threshold=2,
                                   number_in_cadence=6,
                                   user_validation=False,
                                   saving=True,
                                   csv_name=PATH_CSVF)
    print('test_plotting: make_all_dat_files() df_event:\n{}'.format(df_event))

    # Make a list of the HDF5 files.
    print('test_plotting: making a list of HDF5 files in ({}) ...'
          .format(PATH_DAT_LIST_FILE))
    with open(PATH_H5_LIST_FILE, 'w') as file_handle:
        for filename_h5 in FILE_LIST:
            file_handle.write('{}\n'.format(TESTDIR + filename_h5))

    # Do the plots for all of the HDF5/DAT file pairs.
    print('test_plotting: plot_event_pipeline({}, {}) ...'
          .format(PATH_CSVF, PATH_H5_LIST_FILE))
    plot_event_pipeline(PATH_CSVF,
                        PATH_H5_LIST_FILE,
                        user_validation=False)

    main_time_stop = time()
    print('test_plotting: End, et = {:.1f} seconds'
          .format(main_time_stop - main_time_start))


if __name__ == '__main__':
    test_plotting(FLAG_PROGRESS_BAR)
