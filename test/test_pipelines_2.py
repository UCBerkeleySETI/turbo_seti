r'''
Package turbo_seti
test_pipelines_2.py - Expand coverage for individual files in the
                      turbo_seti/find_event folder, especially pipelines.
'''

from time import time
from shutil import rmtree
from pathlib import Path
from os import remove, listdir
import imghdr
import glob
from tempfile import gettempdir
import sys
import numpy as np
import pandas as pd

from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline

TESTDIR = gettempdir() + '/pipeline_testing/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files_2.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files_2.lst'
PATH_CSVF = TESTDIR + 'found_event_table_2.csv'
CSV_DELIM = ','

EXP_SOURCE = ['VOYAGER-1'] * 6
EXP_TOPHITNUM = [1, 1, 1, 2, 2, 2]
EXP_STATUS = ['on_table_1', 'on_table_2', 'on_table_3',
              'on_table_1', 'on_table_2', 'on_table_3']
EXP_CHANINDX = [651879, 651964, 652058, 659989, 660074, 660166]
SNR_LOW = np.array([21.0, 10.0, 17.0, 185.0, 78.0, 140.0])
SNR_HIGH = np.array([24.0, 12.0, 20.0, 199.0, 85.0, 148.0])



def validate_csvf(arg_csvf):
    r'''
    Validate the given CSV file.

    Read in the CSV file into a raw Pandas DataFrame.
    Check that specific columns have the expected values:
      Source, TopHitNum, status, ChanIndx, and SNR.
    Return the Pandas dataframe.
    '''
    df = pd.read_csv(arg_csvf, sep=CSV_DELIM)
    #df.drop('Unnamed: 0')
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
    return df['SNR'].values


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


def find_plot_pipelines(filter_threshold=3,
                        on_off_first='ON',
                        on_source_complex_cadence=False):
    r'''
    Exercise find_event_pipeline() and plot_event_pipeline()
    '''

    main_time_start = time()

    print('find_plot_pipelines: Filter threshold = ', filter_threshold)
    h5_file_list = sorted(glob.glob(TESTDIR + '*.h5'))
    dat_file_list = sorted(glob.glob(TESTDIR + '*.dat'))
    number_in_cadence = len(h5_file_list)
    if number_in_cadence != 6:
        raise ValueError('find_plot_pipelines: Expected to find 6 h5 files but observed {}'
                         .format(number_in_cadence))
    for ix in [0, 2, 4]:
        temp = h5_file_list[ix]
        h5_file_list[ix] = h5_file_list[ix + 1]
        h5_file_list[ix + 1] = temp
        temp = dat_file_list[ix]
        dat_file_list[ix] = dat_file_list[ix + 1]
        dat_file_list[ix + 1] = temp
    fh_h5 = open(PATH_H5_LIST_FILE, 'w')
    fh_dat = open(PATH_DAT_LIST_FILE, 'w')
    for ix in range(6):
        fh_h5.write(h5_file_list[ix] + '\n')
        fh_dat.write(dat_file_list[ix] + '\n')
    fh_h5.close()
    fh_dat.close()
    print('find_plot_pipelines: H5/dat cadence length = ', number_in_cadence)

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
                                   on_off_first=on_off_first,
                                   on_source_complex_cadence=on_source_complex_cadence,
                                   csv_name=PATH_CSVF)

    # CSV file created?
    if not Path(PATH_CSVF).exists():
        raise ValueError('find_plot_pipelines: No CSV of events created')

    # An event CSV was created.
    # Validate CSV file.
    snr_validate = validate_csvf(PATH_CSVF)
    snr_event = df_event['SNR'].values
    print('\n*** snr_event:\n', snr_event)
    print('\n*** snr_validate:\n', snr_validate, '\n')
    if not np.all(snr_validate == snr_event):
        print('\n*** snr_event:\n', snr_event)
        print('\n*** snr_validate:\n', snr_validate, '\n')
        raise ValueError('find_plot_pipelines: df_validate != df_event')

    # Do the plots for all of the HDF5/DAT file pairs.
    png_file_list = sorted(glob.glob(TESTDIR + '*.png'))
    for png_file in png_file_list:
        remove(png_file)
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


def test_pipelines(cleanup=True):
    r'''
    This is the pytest entry point.
    Test filter threshold 3 in find_plot_pipelines().
    By default (pytest):
    * Initialization is done only once.
    * Cleanup is not performed at end.
    '''
    print('\n===== test_pipelines_2: BEGIN =====')

    find_plot_pipelines(filter_threshold=3,
                        on_off_first='OFF',
                        on_source_complex_cadence=False)

    if cleanup:
        rmtree(TESTDIR, ignore_errors=True)

    print('\n===== test_pipelines_2: END =====')


if __name__ == '__main__':
    test_pipelines(cleanup=False)
