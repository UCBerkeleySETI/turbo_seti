r'''
Package turbo_seti
test_pipelines.py - Expand coverage for individual files in the
                    turbo_seti/find_event folder, especially pipelines.

Date         Who            Description
----------   -------------  --------------------------------------------
2020-07-30   R. Elkins      Initial version as 'test_pipelines.py'
2020-08-01   R. Elkins      Renamed to 'test_pipelines.py'
2020-08-01   R. Elkins      Expand filter threshold coverage to 1, 2, & 3
2020-08-18   R. Elkins      Fix test_pipelines execution to re-clean-up
                            between find_plot_pipelines() executions.
2020-12-12   R. Elkins      Issue #127 - use new filter_spec parameter.
2021-01-05   R. Elkins      Re-work to use Voyager 2020 files.
'''

from time import time
from shutil import rmtree
from pathlib import Path
from os import mkdir, remove, listdir
from tempfile import gettempdir
import sys
from urllib.error import HTTPError
from argparse import ArgumentParser
import wget
import pandas as pd
import numpy as np
import imghdr

from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline

TESTDIR = gettempdir() + '/pipeline_testing/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files.lst'
PATH_CSVF = TESTDIR + 'found_event_table.csv'
CSV_DELIM = ','
EXP_SOURCE = ['VOYAGER-1'] * 6
EXP_TOPHITNUM = [1, 1, 1, 2, 2, 2]
EXP_STATUS = ['on_table_1', 'on_table_2', 'on_table_3',
              'on_table_1', 'on_table_2', 'on_table_3']
EXP_CHANINDX = [651879, 651964, 652058, 659989, 660074, 660166]
SNR_LOW = np.array([21.0, 10.0, 17.0, 185.0, 78.0, 140.0])
SNR_HIGH = np.array([24.0, 12.0, 20.0, 199.0, 85.0, 148.0])


URL_DIR = 'http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/'
H5_FILE_LIST = ['single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5',
                'single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5',
                'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5',
                'single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5',
                'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5',
                'single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5']


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


def initialize():
    r'''
    Recreate working directory, TESTDIR.
    '''
    rmtree(TESTDIR, ignore_errors=True)
    mkdir(TESTDIR)
    print('test_pipelines: Initialized')


def wgetter(arg_h5_name):
    r'''
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
    r'''
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
                          max_drift=4,
                          snr=10,
                          out_dir=TESTDIR)
    time_stop = time()
    print('make_one_dat_file: End FindDoppler({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))

    print('make_one_dat_file: Begin Doppler search({}) .....'
          .format(arg_h5_name))

    # ----------------------------------------------------------------------------
    # No more than 1 execution of this program because of dask methodology!
    # To do multiple dask partitions would cause initialization & cleanup chaos.
    time_start = time()
    doppler.search(n_partitions=1)
    time_stop = time()
    # ----------------------------------------------------------------------------

    print('make_one_dat_file: End Doppler search({}), et = {:.1f} seconds'
          .format(arg_h5_name, time_stop - time_start))


def make_all_dat_files():
    r'''
    For each HDF5 file name,
    * Make one DAT file.
    * Add its name to the list of DAT files.
    '''
    with open(PATH_DAT_LIST_FILE, 'w') as file_handle:
        for filename_h5 in H5_FILE_LIST:
            make_one_dat_file(filename_h5)
            filename_dat = filename_h5.replace('.h5', '.dat')
            file_handle.write('{}\n'.format(TESTDIR + filename_dat))


def validate_csvf(arg_csvf):
    r'''
    Validate the given CSV file.
    '''
    # Read in the CSV file into a raw Pandas DataFrame.
    # Check that specific columns have the expected values:
    #   Source, TopHitNum, status, ChanIndx, and SNR.
    # Return the Pandas dataframe.
    df = pd.read_csv(arg_csvf, sep=CSV_DELIM)
    nrows = len(df)
    if nrows != 6:
        raise ValueError('validate_csvf: Expected 6 rows but observed {} rows'
                         .format(nrows))

    obs_source = df['Source'].tolist()
    obs_tophitnum = df['TopHitNum'].tolist()
    obs_status = df['status'].tolist()
    obs_chanindx = df['ChanIndx'].tolist()
    obs_snr = df['SNR'].tolist()
    if obs_source != EXP_SOURCE:
        raise ValueError('validate_csvf: Expected source column {} but observed {}'
                         .format(EXP_SOURCE, obs_source))
    if obs_tophitnum != EXP_TOPHITNUM:
        raise ValueError('validate_csvf: Expected TopHitNum column {} but observed {}'
                         .format(EXP_TOPHITNUM, obs_tophitnum))
    if obs_status != EXP_STATUS:
        raise ValueError('validate_csvf: Expected status column {} but observed {}'
                         .format(EXP_STATUS, obs_status))
    if obs_chanindx != EXP_CHANINDX:
        raise ValueError('validate_csvf: Expected channel index column {} but observed {}'
                         .format(EXP_CHANINDX, obs_chanindx))
    if np.any(obs_snr > SNR_HIGH) or np.any(obs_snr < SNR_LOW):
        raise ValueError('validate_csvf: Expected SNR column in range of {}:{} but observed {}'
                         .format(SNR_LOW, SNR_HIGH, obs_snr))
    return df


def find_plot_pipelines(need_init=True, filter_threshold=3):
    r'''
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

    # CSV file created?
    if not Path(PATH_CSVF).exists():
        raise ValueError('find_plot_pipelines: No CSV of events created')

    # An event CSV was created.
    # Validate CSV file.
    df_validate = validate_csvf(PATH_CSVF)
    if df_validate.equals(df_event):
        raise ValueError('find_plot_pipelines: Expected df_validate to be identical to df_event but observed\n{}'
                         .format(df_validate))

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
                        filter_spec='f{}'.format(filter_threshold),
                        user_validation=False)

    # Check that the right number of PNG files were created.
    outdir_list = listdir(TESTDIR)
    npngs = 0
    for cur_file in outdir_list:
        if cur_file.split('.')[-1] == 'png':
            if imghdr.what(TESTDIR + cur_file) != 'png':
                raise ValueError('find_plot_pipelines: File {} is not a PNG file'
                                 .format(cur_file))
            npngs += 1
    if npngs != 6:
        raise ValueError('find_plot_pipelines: Expected to find 6 PNG files but observed {}'
                         .format(npngs))

    # Stop the clock - we're done.
    main_time_stop = time()

    print('find_plot_pipelines: End, et = {:.1f} seconds'
          .format(main_time_stop - main_time_start))


def test_pipelines(need_init=True, cleanup=False):
    r'''
    This is the pytest entry point.
    Test filter threshold 3 in find_plot_pipelines().
    By default (pytest):
    * Initialization is done only once.
    * Cleanup is not performed at end.
    '''
    print('\n===== test_pipelines: BEGIN =====')

    find_plot_pipelines(need_init=need_init, filter_threshold=3)

    if cleanup:
        rmtree(TESTDIR, ignore_errors=True)

    print('\n===== test_pipelines: END =====')


def main(args=None):
    r'''Main Function Entry Point'''
    pobj = ArgumentParser(description='Test find|plot pipelines.')
    pobj.add_argument('-i', '--initialize', dest='flag_init', type=str,
                   help='Initialize/download? (y/n) - no default')
    pobj.add_argument('-c', '--cleanup', dest='flag_cleanup', type=str,
                   help='Cleanup afterwards? (y/n) - no default')
    if args is None:
        args = pobj.parse_args()
    else:
        args = pobj.parse_args(args)
    test_pipelines(need_init=(args.flag_init == 'y'),
                   cleanup=(args.flag_cleanup == 'y'))


if __name__ == '__main__':
    main()
