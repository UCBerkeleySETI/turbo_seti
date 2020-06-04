import blimpy as bl
from turbo_seti import FindDoppler, seti_event
from turbo_seti import find_event, plot_event
import pylab as plt
import numpy as np
import os
import pytest

HERE = os.path.split(os.path.abspath(__file__))[0]


def find_doppler(filename_fil):
    """ Run turboseti doppler search on filename with default params """
    print("Searching %s" % filename_fil)
    filename_dat = filename_fil.replace('.h5', '.dat')
    filename_log = filename_fil.replace('.h5', 'log')

    if os.path.exists(filename_dat):
        os.remove(filename_dat)

    if os.path.exists(filename_log):
        os.remove(filename_log)

    snr           = 5
    coarse_chans  = ''
    obs_info      = None
    n_coarse_chan = 1
    max_drift     = 1.0

    find_seti_event = FindDoppler(filename_fil, max_drift=max_drift, snr=snr, out_dir=HERE,
                                  coarse_chans=coarse_chans, obs_info=obs_info, n_coarse_chan=n_coarse_chan)
    find_seti_event.search()


def plot_hits(filename_fil, filename_dat):
    """ Plot the hits in a .dat file. """
    table = find_event.read_dat(filename_dat)
    print(table)

    plt.figure(figsize=(10, 8))
    N_hit = len(table)
    if N_hit > 10:
        print("Warning: More than 10 hits found. Only plotting first 10")
        N_hit = 10

    for ii in range(N_hit):
        plt.subplot(N_hit, 1, ii+1)
        plot_event.plot_hit(filename_fil, filename_dat, ii)
    plt.tight_layout()
    plt.savefig(filename_dat.replace('.dat', '.png'))
    plt.show()


def validate_voyager_hits(filename_dat):
    """ This checks voyager hits against known values.

    Known values:
    # --------------------------
    # ID  Drift_Rate SNR 	      Unc_Freq 	      Corr_Freq 	Index   	freq_start 	   freq_end 	...
    # --------------------------
    001	 -0.392226	 30.612128	   8419.319368	   8419.319368	739933	   8419.319779	   8419.318963	...
    002	 -0.373093	245.707984	   8419.297028	   8419.297028	747929	   8419.297439	   8419.296623	...
    003	 -0.392226	 31.220652	   8419.274374	   8419.274374	756037	   8419.274785	   8419.273969	...
    (from flipped)
    003	 -0.392226	 30.612118	   8419.319366	   8419.319366	308642	   8419.318955	   8419.319771	...
    002	 -0.373093	245.707905	   8419.297025	   8419.297025	300646	   8419.296614	   8419.297430	...
    001	 -0.392226	 31.220642	   8419.274372	   8419.274372	292538	   8419.273961	   8419.274777	...
    """
    h = find_event.read_dat(filename_dat)

    valid_data = [
        {
            'Freq': 8419.319368,
            'FreqStart': 8419.319779,
            'FreqEnd': 8419.318963,
            'SNR': 30.612128,
            'DriftRate': -0.392226,
        },
        {
            'Freq': 8419.297028,
            'FreqStart': 8419.297439,
            'FreqEnd': 8419.296623,
            'SNR': 245.707984,
            'DriftRate': -0.373093,
        },
        {
            'Freq': 8419.274374,
            'FreqStart': 8419.274785,
            'FreqEnd': 8419.273969,
            'SNR': 31.220652,
            'DriftRate': -0.392226,
        }
    ]

    atols = {'Freq': 0.000005, 'FreqStart': 0.00001, 'FreqEnd': 0.00001, 'SNR': 0.001, 'DriftRate': 0.02}

    for vd in valid_data:
        hmax = h[np.isclose(h['Freq'], vd['Freq'], rtol=0.000001)].iloc[0]

        for key in vd.keys():
            print(key, hmax[key], vd[key])
            if key in ('FreqStart', 'FreqEnd'):
                upper = np.isclose(hmax[key], vd['FreqStart'], atol=atols[key], rtol=0)
                lower = np.isclose(hmax[key], vd['FreqEnd'], atol=atols[key], rtol=0)
                assert upper or lower
            else:
                assert np.isclose(hmax[key], vd[key], atol=atols[key], rtol=0)
    return hmax


def test_find_doppler_voyager():
    """ Run turboseti on Voyager data """
    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.h5')
    filename_dat = filename_fil.replace('.h5', '.dat')
    find_doppler(filename_fil)
    validate_voyager_hits(filename_dat)
    plot_hits(filename_fil, filename_dat)


def test_find_doppler_voyager_flipped():
    """ Run turboseti on Voyager data (flipped in frequency) """
    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.flipped.h5')
    filename_dat = filename_fil.replace('.h5', '.dat')
    find_doppler(filename_fil)
    validate_voyager_hits(filename_dat)
    plot_hits(filename_fil, filename_dat)

def test_find_doppler_voyager_filterbank():
    """ Run turboseti on Voyager data (filterbank version) """
    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.fil')
    filename_dat = filename_fil.replace('.fil', '.dat')
    find_doppler(filename_fil)
    #validate_voyager_hits(filename_dat)
    #plot_hits(filename_fil, filename_dat)


def test_turboSETI_entry_point():
    """ Test the command line utility turboSETI """

    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.flipped.h5')
    args = [filename_fil, ]
    seti_event.main(args)

def test_plotting():
    """ Some basic plotting tests

    TODO: Improve these tests (and the functions for that matter!
    """
    filename_fil = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.h5')
    fil = bl.Waterfall(filename_fil)

    # Test make_waterfall_plots -- needs 6x files
    filenames_list = [filename_fil] * 6
    target  = 'Voyager'
    drates  = [-0.392226]
    fvals   = [8419.274785]
    f_start = 8419.274374 - 600e-6
    f_stop  = 8419.274374 + 600e-6
    node_string = 'test'
    filter_level = 1
    plot_event.make_waterfall_plots(filenames_list, target, drates, fvals, f_start, f_stop, node_string, filter_level)
    plt.show()

def test_data_handler():
    """ Basic data handler test """
    from turbo_seti.find_doppler import data_handler
    with pytest.raises(AttributeError):
        fh = data_handler.DATAHandle(filename='made_up_not_existing_file.h5')

if __name__ == "__main__":

    #test_turboSETI_entry_point()
    #test_find_doppler_voyager()
    #test_find_doppler_voyager_flipped()
    #test_plotting()
    #test_find_doppler_voyager_filterbank()
    test_data_handler()
