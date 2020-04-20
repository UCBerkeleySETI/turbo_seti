from turbo_seti.find_doppler import FindDoppler
import logging
import sys
import os

HERE = os.path.split(os.path.abspath(__file__))[0]

def test_find_doppler_voyager():

    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.h5')
    filename_dat = filename_fil.replace('.h5', '.dat')
    filename_log = filename_fil.replace('.h5', 'log')

    if os.path.exists(filename_dat):
        os.remove(filename_dat)

    if os.path.exists(filename_log):
        os.remove(filename_log)

    snr           = 10
    coarse_chans  = ''
    obs_info      = None
    n_coarse_chan = 1
    max_drift     = 1.0

    level_log = logging.INFO
    format = '%(name)-15s %(levelname)-8s %(message)s'
    stream = sys.stdout
    logging.basicConfig(format=format,stream=stream,level = level_log)

    find_seti_event = FindDoppler(filename_fil, max_drift=max_drift, snr=snr, out_dir=HERE,
                                  coarse_chans=coarse_chans, obs_info=obs_info, n_coarse_chan=n_coarse_chan)
    find_seti_event.search()

if __name__ == "__main__":
    test_find_doppler_voyager()