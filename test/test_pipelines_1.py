r'''
test_pipelines.py

Using the 0000.h5 Voyager 2020 set of HDF5 files
from http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/
test the following:
* FindDoppler seaarch
* find_event_pipeline
* plot_event_pipeline
'''

from time import time
from shutil import rmtree
from pathlib import Path
from os import mkdir, listdir
from tempfile import gettempdir
import sys
from urllib.error import HTTPError
from argparse import ArgumentParser
import imghdr
import wget
import pytest

from turbo_seti.find_doppler.find_doppler import FindDoppler
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
import pipelines_util as utl

TESTDIR = gettempdir() + '/pipeline_testing/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files.lst'
PATH_CSVF = TESTDIR + 'found_event_table.csv'

URL_DIR = 'http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/'
H5_FILE_LIST = ['single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5',
                'single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5',
                'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5',
                'single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5',
                'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5',
                'single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5']
MAX_DRIFT = 2.0
MIN_SNR = 10.0


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
    print('test_pipelines_1: Initialized')


def wgetter(arg_h5_name):
    r'''
    wget an HDF5 file from the Internet repository.
    arg_h5_name:  HDF5 file name
    '''
    url_h5 = URL_DIR + arg_h5_name
    path_h5 = TESTDIR + arg_h5_name
    print('test_pipelines_1: Begin wget {} -> {} .....'.format(url_h5, path_h5))
    time_start = time()
    try:
        wget.download(url_h5, path_h5, bar=False)
    except HTTPError as ex:
        oops('test_pipelines_1: wget {}, failed: {}'.format(url_h5, repr(ex)))
    time_stop = time()
    print('test_pipelines_1: End wget ({}), et = {:.1f} seconds'
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
                          max_drift=MAX_DRIFT,
                          snr=MIN_SNR,
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


def find_plot_pipelines(need_init=True, filter_threshold=3):
    r'''
    Exercise find_event_pipeline() and plot_event_pipeline()
    '''

    main_time_start = time()

    # If configured to do so, initialize temp directory
    # and fetch all of the HDF5 files from the Internet.
    if need_init:
        initialize()
        for filename_h5 in H5_FILE_LIST:
            wgetter(filename_h5)

    # Make all of the DAT files.
    make_all_dat_files()

    print('find_plot_pipelines: Filter threshold = ', filter_threshold)
    number_in_cadence = len(H5_FILE_LIST)
    print('find_plot_pipelines: Cadence length = ', number_in_cadence)
    print('find_plot_pipelines: find_event_pipeline({}) ...'
          .format(PATH_DAT_LIST_FILE))

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
    # Validate the hit table file.
    utl.validate_hittbl(df_event, PATH_CSVF, 'test_pipe_lines')

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


@pytest.mark.order(1)
def test_pipelines(need_init=True, cleanup=False):
    r'''
    This is the pytest entry point.
    Test filter threshold 3 in find_plot_pipelines().
    By default (pytest):
    * Initialization is done only once.
    * Cleanup is not performed at end.
    '''
    print('\n===== test_pipelines_1: BEGIN =====')

    find_plot_pipelines(need_init=need_init, filter_threshold=3)

    if cleanup:
        rmtree(TESTDIR, ignore_errors=True)

    print('\n===== test_pipelines_1: END =====')


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
                   cleanup=(args.flag_cleanup == 'n'))


if __name__ == '__main__':
    main()
