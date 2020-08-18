'''
Package turbo_seti
test_pipelines.py - Expand coverage for individual files in the
                    turbo_seti/find_event folder, especially pipelines.

Date         Who            Description
----------   -------------  --------------------------------------------
2020-07-30   R. Elkins      Initial version as "test_pipelines.py"
2020-08-01   R. Elkins      Renamed to "test_pipelines.py"
2020-08-01   R. Elkins      Expand filter threshold coverage to 1, 2, & 3
2020-08-18   R. Elkins      Fix test_pipelines execution to re-clean-up
                            between find_plot_pipelines() executions.
'''

from time import time
from shutil import rmtree
from pathlib import Path
from os import mkdir, remove, listdir
from tempfile import gettempdir
import sys
import os
from urllib.error import HTTPError
import wget

from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline


TESTDIR = gettempdir() + '/pipeline_testing/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files.lst'
PATH_CSVF = TESTDIR + 'found_event_table.csv'


URL_DIR = 'http://blpd0.ssl.berkeley.edu/parkes_testing/'
H5_FILE_LIST = ['diced_Parkes_57941_12846_HIP33499_S_fine.h5',
                'diced_Parkes_57941_13194_HIP33499_R_fine.h5',
                'diced_Parkes_57941_13542_HIP33499_S_fine.h5',
                'diced_Parkes_57941_13884_HIP33499_R_fine.h5',
                'diced_Parkes_57941_14233_HIP33499_S_fine.h5',
                'diced_Parkes_57941_14584_HIP33499_R_fine.h5']


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


def initialize():
    '''
    Recreate working directory, TESTDIR.
    '''
    rmtree(TESTDIR, ignore_errors=True)
    mkdir(TESTDIR)
    print('test_pipelines: Initialized')


def wgetter(arg_h5_name):
    '''
    wget an HDF5 file from the Internet repository.
    arg_h5_name:  HDF5 file name
    '''
    url_h5 = URL_DIR + arg_h5_name
    path_h5 = TESTDIR + arg_h5_name
    print('test_pipelines: Begin wget {} -> {} .....'.format(url_h5, path_h5))
    time_start = time()
    try:
        wget.download(url_h5, path_h5, bar=False)
    except HTTPError as ex:
        oops('test_pipelines: wget {}, failed: {}'.format(url_h5, repr(ex)))
    time_stop = time()
    print('test_pipelines: End wget ({}), et = {:.1f} seconds'
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
    print('make_one_dat_file: Begin FindDoppler({}) .....'.format(arg_h5_name))
    h5_path = TESTDIR + arg_h5_name
    time_start = time()
    doppler = FindDoppler(h5_path,
                          max_drift=1,
                          min_drift=-1,
                          snr=25,
                          out_dir=TESTDIR)
    time_stop = time()
    print('make_one_dat_file: End FindDoppler({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))

    print('make_one_dat_file: Begin Doppler search({}) .....'
          .format(arg_h5_name))
    
    # ----------------------------------------------------------------------------
    # No more than 1 execution of this program because of dask methodology!
    # To do multiple dask partitions, would cause initialization & cleanup chaos.
    time_start = time()
    doppler.search(n_partitions=1)
    time_stop = time()
    # ----------------------------------------------------------------------------
    
    print('make_one_dat_file: End Doppler search({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))


def make_all_dat_files():
    '''
    For each HDF5 file name,
    * Make one DAT file.
    * Add its name to the list of DAT files.
    '''
    with open(PATH_DAT_LIST_FILE, 'w') as file_handle:
        for filename_h5 in H5_FILE_LIST:
            make_one_dat_file(filename_h5)
            filename_dat = filename_h5.replace('.h5', '.dat')
            file_handle.write('{}\n'.format(TESTDIR + filename_dat))


def find_plot_pipelines(need_init=True, filter_threshold=2):
    '''
    Exercise find_event_pipeline() and plot_event_pipeline()
    '''

    main_time_start = time()

    # If configured to do so, initialize temp directory.
    if need_init:
        initialize()

    # If configured to do so, fetch all of the HDF5 files from the Internet.
    if need_init:
        for filename_h5 in H5_FILE_LIST:
            wgetter(filename_h5)

    # Make all of the DAT files.
    make_all_dat_files()

    # Create the CSV file used by plot_event_pipeline.
    print('find_plot_pipelines: Filter threshold = ', filter_threshold)
    number_in_cadence = len(H5_FILE_LIST)
    print('find_plot_pipelines: Cadence length = ', number_in_cadence)
    print('find_plot_pipelines: find_event_pipeline({}) ...'
          .format(PATH_DAT_LIST_FILE))

    # If CSV exists from a previous execution, remove it.
    try:
        remove(PATH_CSVF)
    except:
        pass
    
    # With the list of DAT files, do find_event_pipeline()
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   filter_threshold=filter_threshold,
                                   number_in_cadence=number_in_cadence,
                                   user_validation=False,
                                   saving=True,
                                   csv_name=PATH_CSVF)

    # CSV created?
    if not Path(PATH_CSVF).exists():
        print('*** find_plot_pipelines: No CSV of events created')
        return

    # An event CSV was created.
    print('find_plot_pipelines: find_event_pipeline() returned dataframe:\n{}'.format(df_event))

    # Make a list of the HDF5 files.
    print('find_plot_pipelines: making a list of HDF5 files in ({}) ...'
          .format(PATH_DAT_LIST_FILE))
    with open(PATH_H5_LIST_FILE, 'w') as file_handle:
        for filename_h5 in H5_FILE_LIST:
            file_handle.write('{}\n'.format(TESTDIR + filename_h5))

    # Do the plots for all of the HDF5/DAT file pairs.
    print('find_plot_pipelines: plot_event_pipeline({}, {}) ...'
          .format(PATH_CSVF, PATH_H5_LIST_FILE))
    plot_event_pipeline(PATH_CSVF,
                        PATH_H5_LIST_FILE,
                        user_validation=False)

    main_time_stop = time()
    
    print('find_plot_pipelines: End, et = {:.1f} seconds'
          .format(main_time_stop - main_time_start))


def re_clean():
    for x_file in sorted(listdir(TESTDIR)):
        x_type = x_file.split('.')[-1]
        if x_type != 'h5':
            os.remove(TESTDIR + x_file)


def test_pipelines(need_init=True, cleanup=True):
    '''
    Main testing procedure:
    Test each filter threshold in find_event/find_events().
    By default (unattended testing):
    * Initialization is done only once.
    * Cleanup is performed at end.
    '''
    print("\n===== test_pipelines: BEGIN =====")
    
    find_plot_pipelines(need_init=need_init, filter_threshold=1)
    re_clean()
    find_plot_pipelines(need_init=False, filter_threshold=2)
    re_clean()
    find_plot_pipelines(need_init=False, filter_threshold=3)
    
    if cleanup:
        rmtree(TESTDIR, ignore_errors=True)
    
    print("\n===== test_pipelines: END =====")


if __name__ == '__main__':
    # When run manually, no initialization nor cleanup is performed.
    re_clean()
    test_pipelines(need_init=False, cleanup=False)
