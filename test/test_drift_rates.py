r''' test_drift_rates.py
Test the FindDoppler min_drift and max_drift parameters.
'''

import os
from tempfile import gettempdir
from astropy import units as u
import pandas as pd
import setigen as stg
from fb_cases_util import make_one_dat_file


def gen_fil(arg_path):
    r''' Generate a Filterbank file '''
    
    # Define time and frequency arrays, essentially labels for the 2D data array
    fchans = 1048576
    tchans = 16
    df = 1.0*u.Hz
    dt = 1.0*u.s
    fch1 = 6095.214842353016*u.MHz
    noise_std = 0.05 # Gaussian standard deviation
    
    sig_snr_1 = 100.0
    sig_width_1 = 1.1 * u.Hz
    drate_1 = 1.6 * u.Hz/u.s
    f_start_1 = 0
    
    sig_snr_2 = 200.0
    sig_width_2 = 1.2 * u.Hz
    drate_2 = 1.3 * u.Hz/u.s
    f_start_2 = fchans * 0.1
    
    sig_snr_3 = 300.0
    sig_width_3 = 1.3 * u.Hz
    drate_3 = 2.6 * u.Hz/u.s
    f_start_3 = fchans * 0.2
    
    sig_snr_4 = 400.0
    sig_width_4 = 1.4 * u.Hz
    drate_4 = 3.2 * u.Hz/u.s
    f_start_4 = fchans * 0.3
    
    # Generate the frame.
    frame = stg.Frame(fchans=fchans,
                      tchans=tchans,
                      df=df,
                      dt=dt,
                      fch1=fch1)
    
    # Add noise.
    frame.add_noise(x_mean=0, x_std=noise_std, noise_type='gaussian')

    # Add signal 1.
    signal_intensity = frame.get_intensity(snr=sig_snr_1)
    frame.add_constant_signal(f_start=frame.get_frequency(f_start_1),
                              drift_rate=drate_1,
                              level=signal_intensity,
                              width=sig_width_1,
                              f_profile_type='gaussian')
    
    # Add signal 2.
    signal_intensity = frame.get_intensity(snr=sig_snr_2)
    frame.add_constant_signal(f_start=frame.get_frequency(f_start_2),
                              drift_rate=drate_2,
                              level=signal_intensity,
                              width=sig_width_2,
                              f_profile_type='gaussian')
    
    # Add signal 3.
    signal_intensity = frame.get_intensity(snr=sig_snr_3)
    frame.add_constant_signal(f_start=frame.get_frequency(f_start_3),
                              drift_rate=drate_3,
                              level=signal_intensity,
                              width=sig_width_3,
                              f_profile_type='gaussian')
    
    # Add signal 4.
    signal_intensity = frame.get_intensity(snr=sig_snr_4)
    frame.add_constant_signal(f_start=frame.get_frequency(f_start_4),
                              drift_rate=drate_4,
                              level=signal_intensity,
                              width=sig_width_4,
                              f_profile_type='gaussian')
    
    # Save Filterbank file.
    frame.save_fil(arg_path)


def proc_one_dat_file(arg_path_fil, min_drift=0.0, max_drift=4.0):
    r'''Make a DAT file and process all of its entries'''
    path_dat = arg_path_fil.replace('.fil', '.dat')
    make_one_dat_file(arg_path_fil, min_drift=min_drift, max_drift=max_drift, min_snr=25.0, remove_h5=True)
    df = pd.read_csv(path_dat, header=None, sep=r'\s+', engine='python', comment='#')
    for ix in range(len(df)):
        drate = abs(float(df[1][ix]))
        if drate < min_drift or drate > max_drift:
            print('test_drift_rates proc_one_dat_file: Oops, dat:drate={}, min_drift={}, max_drift={}'
                  .format(drate, min_drift, max_drift))
            raise ValueError('Drift rate in DAT file is out of bounds!')
    

def test_drift_rate_samples():
    print("\n===== test_drift_rate_samples BEGIN =====")
    TESTDIR = gettempdir() + '/drift_testing/'
    if not os.path.exists(TESTDIR):
        os.mkdir(TESTDIR)
    PATH_FIL = TESTDIR + 'abc.fil'
    gen_fil(PATH_FIL)
    
    proc_one_dat_file(PATH_FIL, min_drift=0.0, max_drift=4.0)
    proc_one_dat_file(PATH_FIL, min_drift=2.0, max_drift=2.6)
    proc_one_dat_file(PATH_FIL, min_drift=2.0, max_drift=2.2)
    print("\n===== test_drift_rate_samples END =====")

if __name__ == '__main__':
    test_drift_rate_samples()
