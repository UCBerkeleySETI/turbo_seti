import logbook
import turbo_seti
from turbo_seti.find_doppler.find_doppler import hitsearch, MaxVals, tophitsearch
from turbo_seti.find_doppler.file_writers import FileWriter, LogWriter
import numpy as np

# Enable debug logging
turbo_seti.find_doppler.find_doppler.logger.level = logbook.WARNING
turbo_seti.find_doppler.file_writers.logger.level = logbook.DEBUG

def test_hitsearch_fscrunch():
    """ Test fscrunch for hitsearch """
    print("\n===== test_hitsearch_fscrunch =====")
    fscrunch = 16

    # Make a fake spectrum with some nonzero values
    s  = np.zeros(1024)
    s[504] = 100
    s[505] = 100
    s[506] = 100
    s[507] = 100
    s[508] = 100
    s[509] = 100
    s[510] = 100
    s[511] = 100

    s0         = 0      # Start idx
    s1         = len(s) # stop idx
    tdwidth    = len(s) # (fft len == tdwidth)
    hit_thresh = 10     # SNR threshold
    drift_rate = 0      # drift rate

    # Need to create a minimal header
    header     =  {'NAXIS1': len(s), 'FCNTR': 0.5, 'DELTAF': 1.0/1024, 'baryv': 0}

    ## Create dict of MaxVal() objects, one for each fscrunch level
    # Creates
    #{1: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #2: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #4: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #8: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #16: < turbo_seti.find_doppler.find_doppler.MaxVals>

    d_max_val  = {}
    fs = 1
    fscrunch_to_test = []
    while fs <= fscrunch:
        fscrunch_to_test.append(fs)
        max_val = MaxVals()
        max_val.init(tdwidth // fs)
        d_max_val[fs] = max_val
        fs *= 2

    # Run hitsearch
    d_n_hits, d_max_val = hitsearch(s, s0, s1, hit_thresh, drift_rate, header, tdwidth, d_max_val,
                                    reverse=0, fscrunch=fscrunch)

    # Check things look good
    for fs in fscrunch_to_test:
        assert fs in d_n_hits.keys()
        assert fs in d_max_val.keys()

    assert np.max(d_max_val[1].maxsnr)  == 100
    assert np.max(d_max_val[2].maxsnr)  == 200 / np.sqrt(2)
    assert np.max(d_max_val[4].maxsnr)  == 400 / np.power(np.sqrt(2), 2)
    assert np.max(d_max_val[8].maxsnr)  == 800 / np.power(np.sqrt(2), 3)
    assert np.max(d_max_val[16].maxsnr) == 800 / np.power(np.sqrt(2), 4)

    assert d_n_hits[1] == 8
    assert d_n_hits[2] == 4
    assert d_n_hits[4] == 2
    assert d_n_hits[8] == 1
    assert d_n_hits[16] == 1

def test_tophitsearch_fscrunch():
    """ Test fscrunch for tophitsearch """
    print("\n===== test_tophitsearch_fscrunch =====")
    fscrunch = 16

    # Make a fake spectrum with some nonzero values
    s  = np.zeros(1024)
    s[504:511] = 100

    # Make a waterfall dynamic spectrum (matches the collapsed spec s)
    wf = np.zeros((16, 1024))
    wf[:, 504:511] = 100.0 / 16

    s0         = 0      # Start idx
    s1         = len(s) # stop idx
    tdwidth    = len(s) # (fft len == tdwidth)
    hit_thresh = 10     # SNR threshold
    drift_rate = 0      # drift rate
    tsteps     = 16
    fftlen     = len(s)
    obs_length    = 300
    max_drift     = 1

    # Need to create a minimal header
    header     =  {'NAXIS1': len(s), 'FCNTR': 0.5, 'DELTAF': 1e-6, 'DELTAT': 300/16,
                   'SOURCE': 'Fake', 'RA': '00:00:00', 'DEC': '01:02:03', 'MJD': 50000,
                   'baryv': 0, 'coarse_chan': 0}

    obs_info = {'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0,
                'pulsar_stats': np.zeros(6), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0,
                'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}

    ## Create dict of MaxVal() objects, one for each fscrunch level
    # Creates
    #{1: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #2: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #4: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #8: < turbo_seti.find_doppler.find_doppler.MaxVals>
    #16: < turbo_seti.find_doppler.find_doppler.MaxVals>

    d_max_val  = {}
    fs = 1
    fscrunch_to_test = []
    while fs <= fscrunch:
        fscrunch_to_test.append(fs)
        max_val = MaxVals()
        max_val.init(tdwidth // fs)
        d_max_val[fs] = max_val
        fs *= 2

    # Run hitsearch
    d_n_hits, d_max_val = hitsearch(s, s0, s1, hit_thresh, drift_rate, header, tdwidth, d_max_val,
                                    reverse=0, fscrunch=fscrunch)

    logwriter = LogWriter('test.log')
    filewriter = FileWriter('test.dat', header)

    tophitsearch(wf, d_max_val, tsteps, header, tdwidth, fftlen,
                 max_drift, obs_length, logwriter=logwriter, filewriter=filewriter, obs_info=obs_info, fscrunch=fscrunch)

if __name__ == "__main__":
    #test_hitsearch_fscrunch()
    test_tophitsearch_fscrunch()