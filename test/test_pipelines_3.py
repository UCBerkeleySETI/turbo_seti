r''' test_pipelines_3.py

Complex Cadence tests:
* test_pipeline_same_source: result is identical to test_pipeline.
* test_pipeline_wrong_source: nil hit table result.
* test_pipeline_mixeded: result is identical to test_pipeline.
'''

from tempfile import gettempdir
from pathlib import Path
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from fb_cases_util import generate_fil_file, make_one_dat_file
import pipelines_util as utl

TESTDIR = gettempdir() + '/pipeline_testing/'
PATH_CSVF = TESTDIR + 'found_event_table_3.csv'
CSV_DELIM = ','
PATH_DAT_LIST_FILE = TESTDIR + 'dat_files_3.lst'
DAT_LIST_ONS = [TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat'
               ]
PATH_IRRELEVANT_FIL = TESTDIR + 'abc.fil'
FILE_IRRELEVANT_DAT = 'abc.dat'

DAT_LIST_MIXED_1 = [
                TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
               ]

DAT_LIST_MIXED_2 = [
                TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
               ]

DAT_LIST_MIXED_3 = [
                TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT
               ]

DAT_LIST_MIXED_4 = [
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat',
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT
               ]

DAT_LIST_MIXED_5 = [
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + FILE_IRRELEVANT_DAT,
                TESTDIR + 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.dat',
                TESTDIR + 'single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.dat',
               ]

def test_pipeline_same_source():
    print('\n===== test_pipeline_same_source: BEGIN =====')

    # Make the dat list.
    with open(PATH_DAT_LIST_FILE, 'w') as fh:
        for path_dat in DAT_LIST_ONS:
            fh.write('{}\n'.format(path_dat))

    # With the list of DAT files, do find_event_pipeline()
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   filter_threshold=3,
                                   number_in_cadence=3,
                                   user_validation=False,
                                   saving=True,
                                   on_source_complex_cadence='VOYAGER-1',
                                   csv_name=PATH_CSVF)

    # df_event should not be nil.
    if df_event is None:
        raise ValueError('test_pipeline_same_source: returned pandas df is None!')       

    # CSV file created?
    if not Path(PATH_CSVF).exists():
        raise ValueError('test_pipeline_same_source: No CSV of events created')

    # An event CSV was created.
    # Validate the hit table file.
    utl.validate_hittbl(df_event, PATH_CSVF, 'test_pipeline_same_source')
    print('\n===== test_pipeline_same_source: END =====')
 

def test_pipeline_wrong_source():
    print('\n===== test_pipeline_wrong_source: BEGIN =====')

    # Make the dat list.
    with open(PATH_DAT_LIST_FILE, 'w') as fh:
        for path_dat in DAT_LIST_ONS:
            fh.write('{}\n'.format(path_dat))

    # With the list of DAT files, do find_event_pipeline()
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   filter_threshold=3,
                                   number_in_cadence=3,
                                   user_validation=False,
                                   saving=True,
                                   on_source_complex_cadence='VOYAGER-42',
                                   csv_name=PATH_CSVF)

    # df_event should be nil
    if not df_event is None:
        raise ValueError('test_pipeline_wrong_source: returned pandas df has entries but should be nil!')       

    print('\n===== test_pipeline_wrong_source: END =====')


def try_mixed(arg_list, init_needed=True):
    print('\n===== try_mixed: BEGIN =====')
    
    # If init needed, make the off-cadence file and one DAT file.
    if init_needed:
        generate_fil_file(PATH_IRRELEVANT_FIL, -1, -1)
        make_one_dat_file(PATH_IRRELEVANT_FIL, max_drift=10.0, min_snr=20.0, remove_h5=False)
    
    # Make the dat list.
    with open(PATH_DAT_LIST_FILE, 'w') as fh:
        for path_dat in arg_list:
            fh.write('{}\n'.format(path_dat))

    # With the list of DAT files, do find_event_pipeline()
    df_event = find_event_pipeline(PATH_DAT_LIST_FILE,
                                   sortby_tstart=False,
                                   filter_threshold=3,
                                   number_in_cadence=len(arg_list),
                                   user_validation=False,
                                   saving=True,
                                   on_source_complex_cadence='VOYAGER-1',
                                   csv_name=PATH_CSVF)

    # df_event should not be nil.
    if df_event is None:
        raise ValueError('try_mixed: returned pandas df is None!')       

    # CSV file created?
    if not Path(PATH_CSVF).exists():
        raise ValueError('try_mixed: No CSV of events created')

    # An event CSV was created.
    # Validate the hit table file.
    utl.validate_hittbl(df_event, PATH_CSVF, 'test_pipeline_mixed')

    print('\n===== try_mixed: END =====')


def test_pipeline_mixed(init_needed=True):
    print('\n===== test_pipeline_mixed: BEGIN =====')
    try_mixed(DAT_LIST_MIXED_1, init_needed=init_needed)
    try_mixed(DAT_LIST_MIXED_2, init_needed=False)
    try_mixed(DAT_LIST_MIXED_3, init_needed=False)
    try_mixed(DAT_LIST_MIXED_4, init_needed=False)
    try_mixed(DAT_LIST_MIXED_5, init_needed=False)
    print('\n===== test_pipeline_mixed: END =====')

    
if __name__ == '__main__':
    test_pipeline_same_source()
    test_pipeline_wrong_source()
    test_pipeline_mixed(init_needed=False)
