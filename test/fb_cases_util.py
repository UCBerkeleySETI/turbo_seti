r'''
Utility functions for test_fb_cases.py
'''

from os import mkdir, remove
from os.path import dirname
from shutil import rmtree
import logging
import pandas as pd
import numpy as np
import setigen as stg
from turbo_seti.find_doppler.find_doppler import FindDoppler
from fb_cases_def import HERE, DEBUGGING, RTOL_DIFF, TestResultRecord, SetigenParms

DF_REFERENCE = HERE + '/fb_dat_reference.txt'
SEP = r'\s+'


def initialize(arg_dir):
    r'''
    Recreate working directory, TESTDIR.
    Load result reference tables (2).
    '''
    rmtree(arg_dir, ignore_errors=True)
    mkdir(arg_dir)
    df = pd.read_csv(DF_REFERENCE, sep=SEP, engine='python', comment='#')
    nrows = len(df)
    if nrows < 1:
        raise ValueError('initialize: Empty reference table')
    if nrows % 2 != 0:
        raise ValueError('initialize: Reference table row count ({}) is not divisible by 2'
                         .format(nrows))
    if DEBUGGING:
        print('initialize: Test case reference results: \n', df)
    ref_tophit_1 = []
    ref_tophit_2 = []
    jj = 0
    while jj < nrows:
        record = TestResultRecord()
        record.fdir = int(df['fdir'][jj])
        record.drsign = int(df['drsign'][jj])
        record.tophit_id = int(df['tophit'][jj])
        record.drate = float(df['drate'][jj])
        record.snr = float(df['snr'][jj])
        record.freq = float(df['freq'][jj])
        record.index = int(df['index'][jj])
        ref_tophit_1.append(record)
        if DEBUGGING:
            print('initialize: appended for hit_1:\n', record.to_string() )
        jj += 1
        del record
        record = TestResultRecord()
        record.fdir = int(df['fdir'][jj])
        record.drsign = int(df['drsign'][jj])
        record.tophit_id = int(df['tophit'][jj])
        record.drate = float(df['drate'][jj])
        record.snr = float(df['snr'][jj])
        record.freq = float(df['freq'][jj])
        record.index = int(df['index'][jj])
        ref_tophit_2.append(record)
        if DEBUGGING:
            print('initialize: appended for hit_2:\n', record.to_string() )
        jj += 1
    if DEBUGGING:
        print('initialize: {} test cases loaded.'.format(len(ref_tophit_1)))
    return ref_tophit_1, ref_tophit_2


def generate_fil_file(outpath, flag_fascending, flag_sign_drift_rate):
    r'''
    Using setigen, generate a filterbank file.

    Parameters:
        outpath - full path of where to store the resultant filterbank file.
        flag_fascending - use an ascending (+1) or descending (-1) sequence of frequencies
        flag_sign_drift_rate - use a positive (+1) or negative (-1) drift rate
    '''
    if DEBUGGING:
        print('generate_fil_file: flag_fascending={}, flag_sign_drift_rate={}'
              .format(flag_fascending, flag_sign_drift_rate))

    # Set up setigne parameters
    stg_parms = SetigenParms()
    if flag_sign_drift_rate < 0:
        stg_parms.drift_rate_1 = -stg_parms.drift_rate_1
        stg_parms.drift_rate_2 = -stg_parms.drift_rate_2
        stg_parms.drift_rate_3 = -stg_parms.drift_rate_3
        stg_parms.drift_rate_4 = -stg_parms.drift_rate_4
        stg_parms.drift_rate_5 = -stg_parms.drift_rate_5

    # Instantiate a setigen Frame object
    frame = stg.Frame(fchans=stg_parms.fchans,
                      tchans=stg_parms.tchans,
                      df=stg_parms.df,
                      dt=stg_parms.dt,
                      fch1=stg_parms.fch1,
                      ascending=(flag_fascending > 0))

    # Add noise to stg object.
    frame.add_noise(x_mean=0, x_std=stg_parms.noise_std, noise_type='gaussian')

    # Signal 1 will be detected.
    signal_1_intensity = frame.get_intensity(snr=stg_parms.snr_1)
    frame.add_constant_signal(f_start=frame.get_frequency(stg_parms.signal_start_1),
                              drift_rate=stg_parms.drift_rate_1,
                              level=signal_1_intensity,
                              width=stg_parms.width_1,
                              f_profile_type='gaussian')

    # Signal 2 will be detected.
    signal_2_intensity = frame.get_intensity(snr=stg_parms.snr_2)
    frame.add_constant_signal(f_start=frame.get_frequency(stg_parms.signal_start_2),
                              drift_rate=stg_parms.drift_rate_2,
                              level=signal_2_intensity,
                              width=stg_parms.width_2,
                              f_profile_type='gaussian')

    # Signal 3 is a symmetric signal with three Gaussians
    # that will fall below the SNR requirements.
    signal_3_intensity = frame.get_intensity(snr=stg_parms.snr_3)
    frame.add_signal(stg.constant_path(f_start=frame.get_frequency(stg_parms.signal_start_3),
                                       drift_rate=stg_parms.drift_rate_3),
                     stg.constant_t_profile(level=1),
                     stg.multiple_gaussian_f_profile(width=stg_parms.width_3),
                     stg.constant_bp_profile(level=signal_3_intensity))

    # Signal 4 is a symmetric signal with three Gaussians
    # that will be drifting too quickly.
    signal_4_intensity = frame.get_intensity(snr=stg_parms.snr_4)
    frame.add_signal(stg.constant_path(f_start=frame.get_frequency(stg_parms.signal_start_4),
                                       drift_rate=stg_parms.drift_rate_4),
                     stg.constant_t_profile(level=1),
                     stg.multiple_gaussian_f_profile(width=stg_parms.width_4),
                     stg.constant_bp_profile(level=signal_4_intensity))

    # Signal 5 is similar to signal 4 but drifting in the opposite direction.
    signal_5_intensity = frame.get_intensity(snr=stg_parms.snr_5)
    frame.add_signal(stg.constant_path(f_start=frame.get_frequency(stg_parms.signal_start_5),
                                       drift_rate=stg_parms.drift_rate_5),
                     stg.constant_t_profile(level=1),
                     stg.multiple_gaussian_f_profile(width=stg_parms.width_5),
                     stg.constant_bp_profile(level=signal_5_intensity))

    # Save the frame as a filterbank file.
    frame.save_fil(filename=outpath)
    del frame


def make_one_dat_file(arg_path_fil, min_drift=0.0 max_drift=4.0, min_snr=25.0, remove_h5=True):
    r'''
    Make a single DAT file:
    * Instantiate the FindDoppler class object.
    * With the object, search the H5, creating the DAT file
      and a LOG file (not used).
    '''
    if max_drift is None:
        raise ValueError('make_one_dat_file: max_drift not set')
    woutdir = dirname(arg_path_fil)
    fdop = FindDoppler(datafile=arg_path_fil,
                       min_drift=min_drift,
                       max_drift=max_drift,
                       snr=min_snr,
                       log_level_int=logging.WARNING,
                       out_dir=woutdir)
    fdop.search()
    path_h5_file = arg_path_fil.replace('.fil', '.h5')
    if remove_h5:
        remove(path_h5_file)


def get_case_results(arg_path_dat):
    r'''From the DAT file, extract the data for all top hits.'''
    df = pd.read_csv(arg_path_dat, header=None, sep=SEP, engine='python', comment='#')
    nrows = len(df)
    if nrows != 2:
        raise ValueError('get_case_results: Expected 2 rows in DAT but observed {} rows'
              .format(nrows))

    obs_tophit_1 = TestResultRecord()
    obs_tophit_1.tophit_id = int(df[0][0]) # 1st col, 1st row
    obs_tophit_1.drate = float(df[1][0])
    obs_tophit_1.snr = float(df[2][0])
    obs_tophit_1.freq = float(df[4][0])
    obs_tophit_1.index = int(df[5][0])

    obs_tophit_2 = TestResultRecord()
    obs_tophit_2.tophit_id = int(df[0][1]) # 1st col, 2nd row
    obs_tophit_2.drate = float(df[1][1])
    obs_tophit_2.snr = float(df[2][1])
    obs_tophit_2.freq = float(df[4][1])
    obs_tophit_2.index = int(df[5][1])

    return obs_tophit_1, obs_tophit_2


def case_comparison(obs_tophit, ref_tophit, max_drift):
    r'''Compare DAT file observations to the reference.'''
    if obs_tophit is None:
        if ref_tophit is None:
            return # success, both None
        # ref_tophit defined, obs_tophit is None
        raise ValueError('case_comparison: FAILED, max_drift={}\nobs_tophit is None\nref_tophit:::{}'
                             .format(max_drift, ref_tophit.to_string()))
    if ref_tophit is None:  # obs_tophit defined, ref_tophit is None
        raise ValueError('case_comparison: FAILED, max_drift={}\nref_tophit is None\nobs_tophit:::{}'
                             .format(max_drift, obs_tophit.to_string()))

    if obs_tophit.tophit_id == ref_tophit.tophit_id \
    and np.isclose(obs_tophit.drate, ref_tophit.drate, rtol=RTOL_DIFF) \
    and np.isclose(obs_tophit.snr, ref_tophit.snr, rtol=RTOL_DIFF) \
    and np.isclose(obs_tophit.freq, ref_tophit.freq, rtol=RTOL_DIFF) \
    and obs_tophit.index == ref_tophit.index:
        return # success

    # Some field(s) did not compare correctly.
    raise ValueError('case_comparison: FAILED, max_drift={}\nobs_tophit:::{}\nref_tophit:::{}'
                     .format(max_drift, obs_tophit.to_string(), ref_tophit.to_string()))

if __name__ == '__main__':
    # __main__ is a developer unit test, not normally to be executed.
    from fb_cases_def import TESTDIR, PATH_FIL_FILE, MIN_SNR
    rmtree(TESTDIR, ignore_errors=True)
    mkdir(TESTDIR)
    generate_fil_file(PATH_FIL_FILE, -1, -1)
    make_one_dat_file(PATH_FIL_FILE, max_drift=5, min_snr=MIN_SNR)
