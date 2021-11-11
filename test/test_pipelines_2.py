r'''
test_pipelines.py

Same as test_pipelines except that the cadence order is reveresed
OFF-ON-... instead of ON-OFF-...
'''

from time import time
from shutil import rmtree
from pathlib import Path
from os import remove, listdir
import imghdr
import glob
from tempfile import gettempdir
import sys

from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
import pipelines_util as utl

TESTDIR = gettempdir() + '/pipeline_testing/'
PLOTDIR = TESTDIR + 'plots/'
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files_2.lst'
PATH_H5_LIST_FILE = TESTDIR + 'h5_files_2.lst'
PATH_CSVF = TESTDIR + 'found_event_table_2.csv'
N_EVENTS = 2


def oops(arg_text):
    '''
    Log the bad news and exit to the O/S with a non-zero exit code.
    '''
    print('\n*** Oops, ' + arg_text)
    sys.exit(86)


def find_plot_pipelines(filter_threshold=3,
                        on_off_first='ON',
                        plot_dir=None,
                        on_source_complex_cadence=False):
    r'''
    Exercise find_event_pipeline() and plot_event_pipeline()
    '''

    main_time_start = time()

    print('find_plot_pipelines_2: Filter threshold = ', filter_threshold)
    h5_file_list = sorted(glob.glob(TESTDIR + 'single*.h5'))
    dat_file_list = sorted(glob.glob(TESTDIR + 'single*.dat'))
    number_in_cadence = len(h5_file_list)
    if number_in_cadence != 6:
        raise ValueError('find_plot_pipelines_2: Expected to find 6 h5 files but observed {}'
                         .format(number_in_cadence))
    number_in_cadence = len(dat_file_list)
    if number_in_cadence != 6:
        raise ValueError('find_plot_pipelines_2: Expected to find 6 dat files but observed {}'
                         .format(number_in_cadence))

    # Re-order the H5 and DAT files into OFF-ON-...
    # In the 2 lists, switch 1 and 2, 3 and 4, 5 and 6
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
    print('find_plot_pipelines_2: H5/dat cadence length = ', number_in_cadence)

    # If CSV exists from a previous execution, remove it.
    try:
        remove(PATH_CSVF)
    except:
        pass

    # With the list of DAT files, do find_event_pipeline()
    print('===== find_event_pipeline BEGIN =====')
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   sortby_tstart=False,
                                   filter_threshold=filter_threshold,
                                   number_in_cadence=number_in_cadence,
                                   user_validation=False,
                                   saving=True,
                                   on_off_first=on_off_first,
                                   on_source_complex_cadence=on_source_complex_cadence,
                                   csv_name=PATH_CSVF)
    print('===== find_event_pipeline END =====')

    # CSV file created?
    if not Path(PATH_CSVF).exists():
        raise ValueError('find_plot_pipelines_2: No CSV of events created')

    # An event CSV was created.
    # Validate the hit table file.
    utl.validate_hittbl(df_event, PATH_CSVF, 'test_pipe_lines_2', N_EVENTS)

    # Do the plots for all of the HDF5/DAT file pairs.
    print('===== plot_event_pipeline #1 (plot_dir does not yet exist) BEGIN =====')
    rmtree(plot_dir, ignore_errors=True)
    plot_event_pipeline(PATH_CSVF,
                        PATH_H5_LIST_FILE,
                        filter_spec='f{}'.format(filter_threshold),
                        plot_dir=plot_dir,
                        user_validation=False)
    print('===== plot_event_pipeline #2 (plot_dir already exists) BEGIN =====')
    plot_event_pipeline(PATH_CSVF,
                        PATH_H5_LIST_FILE,
                        filter_spec='f{}'.format(filter_threshold),
                        plot_dir=plot_dir,
                        user_validation=False)

    # Check that the right number of PNG files were created.
    print('===== plot_event_pipeline END =====')
    outdir_list = listdir(plot_dir)
    npngs = 0
    for cur_file in outdir_list:
        if cur_file.split('.')[-1] == 'png':
            if imghdr.what(plot_dir + cur_file) != 'png':
                raise ValueError('find_plot_pipelines_2: File {} is not a PNG file'
                                 .format(cur_file))
            npngs += 1
    if npngs != N_EVENTS:
        raise ValueError('find_plot_pipelines_2: Expected to find 6 PNG files but observed {}'
                         .format(npngs))

    # Stop the clock - we're done.
    main_time_stop = time()

    print('find_plot_pipelines_2: End, et = {:.1f} seconds'
          .format(main_time_stop - main_time_start))


def test_pipelines_2(cleanup=False):
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
                        plot_dir=PLOTDIR,
                        on_source_complex_cadence=False)

    if cleanup:
        rmtree(TESTDIR, ignore_errors=True)

    print('\n===== test_pipelines_2: END =====')


if __name__ == '__main__':
    test_pipelines_2(cleanup=False)
