"""
Testing for turbo_seti various functions.
TODO: more description of actual coverage versus ideal coverage.
"""
import os
import time
import tempfile

import pylab as plt
import numpy as np
import pytest

from blimpy import Waterfall

from turbo_seti import FindDoppler, seti_event, find_event, plot_event
from turbo_seti.find_doppler.kernels import Kernels
from turbo_seti.find_doppler.find_doppler import search_coarse_channel
from turbo_seti.find_doppler import data_handler, helper_functions, file_writers

HERE = os.path.split(os.path.abspath(__file__))[0]
VOYAH5 = 'Voyager1.single_coarse.fine_res.h5'
VOYAH5FLIPPED = 'Voyager1.single_coarse.fine_res.flipped.h5'
VOYAFIL = 'Voyager1.single_coarse.fine_res.fil'
OFFNIL_H5 = 'single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5'

TESTS =  [
    (Kernels(gpu_backend=False, precision=2)),
    (Kernels(gpu_backend=False, precision=1)),
]

if Kernels.has_gpu():
    GPU_TESTS = [
        (Kernels(gpu_backend=True, precision=2)),
        (Kernels(gpu_backend=True, precision=1)),
    ]
    TESTS.extend(GPU_TESTS)


def find_doppler(filename_fil, kernels):
    """ Run turboseti doppler search on filename with default params """
    print("\n===== find_doppler =====")
    print("Searching %s" % filename_fil)
    filename_dat = filename_fil.replace('.h5', '.dat')
    filename_log = filename_fil.replace('.h5', 'log')

    if os.path.exists(filename_dat):
        os.remove(filename_dat)

    if os.path.exists(filename_log):
        os.remove(filename_log)

    snr           = 5.0
    coarse_chans  = ''
    obs_info      = None
    n_coarse_chan = 1
    max_drift     = 1.0

    find_seti_event = FindDoppler(filename_fil, max_drift=max_drift, snr=snr, out_dir=HERE,
                                  coarse_chans=coarse_chans, obs_info=obs_info,
                                  n_coarse_chan=n_coarse_chan, kernels=kernels)

    t0 = time.time()
    find_seti_event.search()
    t_taken = time.time() - t0
    print("Time taken for find_seti_event.search() [without flagging]: %2.2fs" % t_taken)

    find_seti_event.flagging = True
    t0 = time.time()
    for dummy, data_dict in enumerate(find_seti_event.data_handle.data_list):
        search_coarse_channel(data_dict, find_seti_event, filewriter=None, logwriter=None)
    t_taken = time.time() - t0
    print("Time taken for search_coarse_channe() [with flagging]: %2.2fs" % t_taken)


def plot_hit(fil_filename, dat_filename, hit_id, bw=None, offset=0):
    """Plot a candidate from a .dat file

    Args:
      fil_filename(str): Path to filterbank file to plot
      dat_filename(str): Path to turbosSETI generated .dat output file of events
      hit_id(int): ID of hit in the dat file to plot (TopHitNum)
      offset(float, optional): Offset drift line on plot. Default 0.
      bw:  (Default value = None)

    Returns:

    """
    # Load hit details
    dat = find_event.make_table(dat_filename)
    hit = dat.iloc[hit_id]

    f0 = hit['Freq']

    if bw is None:
        bw_mhz = np.abs(hit['FreqStart'] - hit['FreqEnd'])
    else:
        bw_mhz = bw * 1e-6

    fil = Waterfall(fil_filename, f_start=f0 - bw_mhz / 2, f_stop=f0 + bw_mhz / 2)
    t_duration = (fil.n_ints_in_file - 1) * fil.header['tsamp']

    fil.plot_waterfall()
    plot_event.overlay_drift(f0, f0, f0, hit['DriftRate'], t_duration, offset)


def plot_hits(filename_fil, filename_dat):
    """ Plot the hits in a .dat file. """
    print("\n===== plot_hits =====")
    table = find_event.read_dat(filename_dat)
    print(table)

    plt.figure(figsize=(10, 8))
    N_hit = len(table)
    if N_hit > 10:
        print("Warning: More than 10 hits found. Only plotting first 10")
        N_hit = 10

    for ii in range(N_hit):
        plt.subplot(N_hit, 1, ii+1)
        plot_hit(filename_fil, filename_dat, ii)
    plt.tight_layout()
    plt.savefig(filename_dat.replace('.dat', '.png'))


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
    print("\n===== validate_voyager_hits =====")
    h = find_event.read_dat(filename_dat)
    print(h)

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


@pytest.mark.parametrize("kernels", TESTS)
def test_find_doppler_voyager(kernels):
    """ Run turboseti on Voyager data """
    print("\n===== test_find_doppler_voyager =====")
    filename_fil = os.path.join(HERE, VOYAH5)
    filename_dat = filename_fil.replace('.h5', '.dat')
    find_doppler(filename_fil, kernels)
    validate_voyager_hits(filename_dat)
    plot_hits(filename_fil, filename_dat)


@pytest.mark.parametrize("kernels", TESTS)
def test_find_doppler_voyager_flipped(kernels):
    """ Run turboseti on Voyager data (flipped in frequency) """
    print("\n===== test_find_doppler_voyager_flipped =====")
    filename_fil = os.path.join(HERE, VOYAH5FLIPPED)
    filename_dat = filename_fil.replace('.h5', '.dat')
    find_doppler(filename_fil, kernels)
    validate_voyager_hits(filename_dat)
    plot_hits(filename_fil, filename_dat)


@pytest.mark.parametrize("kernels", TESTS)
def test_find_doppler_voyager_filterbank(kernels):
    """ Run turboseti on Voyager data (filterbank version) """
    print("\n===== test_find_doppler_voyager_filterbank =====")
    filename_fil = os.path.join(HERE, VOYAH5)
    find_doppler(filename_fil, kernels)


def test_turboSETI_entry_point():
    """ Test the command line utility turboSETI """
    print("\n===== test_turboSETI_entry_point 1 =====")
    h5_1 = os.path.join(HERE, VOYAH5FLIPPED)
    args = [h5_1, ]
    seti_event.main(args)
    print("\n===== test_turboSETI_entry_point 2 =====")
    h5_2 = os.path.join(HERE, OFFNIL_H5)
    args = [h5_2, ]
    seti_event.main(args)
    print("\n===== test_turboSETI_entry_point 3 =====")
    h5_3 = os.path.join(HERE, OFFNIL_H5)
    args = [h5_3, "-l", "debug", ]
    seti_event.main(args)
    print("\n===== test_turboSETI_entry_point 4 =====")
    h5_4 = os.path.join(HERE, OFFNIL_H5)
    args = [h5_4, "-g", "y", ]
    seti_event.main(args)
    print("\n===== test_turboSETI_entry_point 5 =====")
    h5_5 = os.path.join(HERE, OFFNIL_H5)
    args = [h5_5, "-P", "y", ]
    seti_event.main(args)


def test_make_waterfall_plots():
    """ Some basic plotting tests

    TODO: Improve these tests (and the functions for that matter!
    """
    print("\n===== test_plotting =====")
    filename_fil = os.path.join(HERE, VOYAH5)

    # Test make_waterfall_plots -- needs 6x files
    filenames_list = [filename_fil] * 6
    target  = 'Voyager'
    drate   = -0.392226
    fmid   =  8419.274785
    f_start = 8419.274374 - 600e-6
    f_stop  = 8419.274374 + 600e-6
    source_name_list = ['test_make_waterfall_plots'] * 6
    filter_level = "1"
    plot_event.make_waterfall_plots(filenames_list, 
                                    target, 
                                    f_start, 
                                    f_stop, 
                                    drate,
                                    fmid,
                                    filter_level,
                                    source_name_list)


@pytest.mark.parametrize("kernels", TESTS)
def test_data_handler(kernels):
    """ Basic data handler test """
    print("\n===== test_data_handler =====")
    with pytest.raises(IOError):
        data_handler.DATAHandle(filename='made_up_not_existing_file.h5', kernels=kernels)
    with pytest.raises(IOError):
        data_handler.DATAHandle(filename=os.path.abspath(__file__), kernels=kernels)
    filename_fil = os.path.join(HERE, VOYAFIL)
    with pytest.raises(IOError):
        out_dir = os.path.join(tempfile.mkdtemp()) + '/NO/SUCH/DIRECTORY'
        dh = data_handler.DATAHandle(filename=filename_fil,
                                     out_dir=out_dir,
                                     n_coarse_chan=42, 
                                     coarse_chans=None,
                                     kernels=kernels)
    dh = data_handler.DATAHandle(filename=filename_fil,
                                 out_dir=os.path.join(tempfile.mkdtemp()),
                                 n_coarse_chan=42, 
                                 coarse_chans=None,
                                 kernels=kernels)
    assert dh.status
    filename_h5 = os.path.join(HERE, VOYAH5)
    dh = data_handler.DATAHandle(filename=filename_h5, 
                                 n_coarse_chan=42, 
                                 coarse_chans=(8300, 8400),
                                 kernels=kernels)
    assert dh.status


@pytest.mark.parametrize("kernels", TESTS)
def test_dask(kernels):
    """ Test dask capability on Voyager data """
    print("\n===== test_dask ===== begin")
    filename_h5 = os.path.join(HERE, VOYAH5)
    FD = FindDoppler(datafile=filename_h5, max_drift=2, out_dir=HERE, kernels=kernels)
    print("===== test_dask ===== n_partitions=None")
    FD.search()
    print("===== test_dask ===== n_partitions=2")
    FD.search(n_partitions=2)
    print("===== test_dask ===== n_partitions=2, progress_bar='n'")
    FD.search(n_partitions=2, progress_bar='n')
    print("===== test_dask ===== merge resulted in a DAT for both flipped and unflipped H5")
    unflipped_dat = filename_h5.replace('.h5', '.dat')
    filename_h5 = os.path.join(HERE, VOYAH5FLIPPED)
    FD = FindDoppler(datafile=filename_h5, max_drift=2, out_dir=HERE, kernels=kernels)
    FD.search(n_partitions=2)
    flipped_dat = filename_h5.replace('.h5', '.dat')
    assert os.path.exists(unflipped_dat)
    assert os.path.exists(flipped_dat)
    print("===== test_dask ===== End")


@pytest.mark.parametrize("kernels", TESTS)
def test_bitrev(kernels):
    '''compare Python and Numba bitrev functions'''
    print("\n===== test_bitrev")
    before = 32769
    nbits = 7
    out_c = kernels.bitrev(before, nbits)
    out_p = helper_functions.bitrev(before, nbits)
    assert out_c == out_p
    before = 32770
    out_c = kernels.bitrev(before, nbits)
    out_p = helper_functions.bitrev(before, nbits)
    assert out_c == out_p
    before = 32771
    out_c = kernels.bitrev(before, 1)
    out_p = helper_functions.bitrev(before, 1)
    assert out_c == out_p == before


def test_compstats(xp=None):
    tp = np if not xp else xp
    arr = tp.array([1., 1., 1., 1., 50.])
    median, stddev = helper_functions.comp_stats(arr, xp)
    assert tp.isclose(median, 1.)
    assert tp.isclose(stddev, 0.)


def test_flipx(xp=None):
    tp = np if not xp else xp
    buf = tp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    exp = tp.array([5, 4, 3, 2, 1, 10, 9, 8, 7, 6])
    helper_functions.FlipX(buf, 5, 2, xp)
    assert tp.allclose(buf, exp)


@pytest.mark.parametrize("kernels", TESTS)
def test_compstats_kernel(kernels):
    test_compstats(kernels.xp)


@pytest.mark.parametrize("kernels", TESTS)
def test_flipx_kernel(kernels):
    test_flipx(kernels.xp)


if __name__ == "__main__":
    print("Please run: pytest test_turbo_seti.py")