from tempfile import gettempdir
from shutil import rmtree
from os import mkdir
from numpy import isclose
from turbo_seti.find_event.find_event import calc_freq_range, find_events

TESTDIR = gettempdir() + '/test_find_event/'
RTOL_DIFF = 0.001 # numpy.isclose(), 0.1%

# Hits 1-3 in table 1 are in all 3 ON tables and in no OFF tables.
# Hit 4 in table 1 is in one ON table and in no OFF tables.
# Hit 5 in table 1 is in one ON table and in one OFF table.
    
DAT1_TABLE = [
    [1, -0.392226, 30.612128, 8419.319368, 8419.319368, 739933, 8419.321003, 8419.317740, 0.0, 0.000000, 0, 858],
    [2, -0.373093, 245.707984, 8419.297028, 8419.297028, 747929, 8419.298662, 8419.295399, 0.0, 0.000000, 0, 858],
    [3, -0.392226, 31.220652, 8419.274374, 8419.274374, 756037, 8419.276009, 8419.272745, 0.0, 0.000000, 0, 858],
    [4, -0.392226, 431.220652, 9419.274374, 9419.274374, 756001, 9419.276009, 8419.272745, 0.0, 0.000000, 0, 858],
    [5, -0.392226, 531.220652, 9319.274374, 9319.274374, 756011, 9319.276009, 8419.272745, 0.0, 0.000000, 0, 858]
]

DAT2_TABLE = [
    [1, 0.0, 4200.0, 10000.0, 10000.0, 739933, 10000.0, 10000.0, 0.0, 0.000000, 0, 858],
    [2, 0.0005, 500.0, 10000.0, 10000.0, 747929, 10000.0, 10000.0, 0.0, 0.000000, 0, 858],
    [3, -2.0005, 700.0, 10000.0, 10000.0, 756037, 10000.0, 10000.0, 0.0, 0.000000, 0, 858],
    [4, -0.382226, 531.220652, 9319.274374, 9319.274374, 756011, 9319.276009, 8419.272745, 0.0, 0.000000, 0, 858]
]

DAT3_TABLE = [
    [1, -0.392226, 30.612128, 8419.319368, 8419.319368, 739933, 8419.321003, 8419.317740, 0.0, 0.000000, 0, 858],
    [2, -0.373093, 245.707984, 8419.297028, 8419.297028, 747929, 8419.298662, 8419.295399, 0.0, 0.000000, 0, 858],
    [3, -0.412226, 31.220652, 8419.274374, 8419.274374, 756037, 8419.276009, 8419.272745, 0.0, 0.000000, 0, 858]
]

DAT4_TABLE = []

DAT5_TABLE = [
    [1, -0.392226, 30.612128, 8419.319368, 8419.319368, 739933, 8419.321003, 8419.317740, 0.0, 0.000000, 0, 858],
    [2, -0.373093, 245.707984, 8419.297028, 8419.297028, 747929, 8419.298662, 8419.295399, 0.0, 0.000000, 0, 858],
    [3, -0.392226, 31.220652, 8419.274374, 8419.274374, 756037, 8419.276009, 8419.272745, 0.0, 0.000000, 0, 858]
]

HEADER_LINES = [
    "# -------------------------- o --------------------------",
    "# File ID: Voyager1.single_coarse.fine_res.h5 ",
    "# -------------------------- o --------------------------",
    "# Source:Voyager1",
    "# MJD: 57650.782094907408	RA: 17h10m03.984s	DEC: 12d10m58.8s",
    "# DELTAT:  18.253611	DELTAF(Hz):  -2.793968	max_drift_rate:   4.000000	obs_length: 292.057776",
    "# --------------------------",
    "# Top_Hit_# 	Drift_Rate 	SNR 	Uncorrected_Frequency 	Corrected_Frequency 	Index 	freq_start 	freq_end 	SEFD 	SEFD_freq 	Coarse_Channel_Number 	Full_number_of_hits",
    "# --------------------------"
]

#=============================== Functions ==============================


def write_one_dat_file(arg_table, arg_path):
    with open(arg_path, "w") as fh:
        for textline in HEADER_LINES:
            fh.write(textline + "\n")
        for hit_entry in arg_table:
            textline = ""
            for item in hit_entry:
                textline += "{}\t".format(item)
            fh.write(textline + "\n")


def write_all_dat_files(dat_table_list, dat_file_list):
    rmtree(TESTDIR, ignore_errors=True)
    mkdir(TESTDIR)
    ix = 0
    for dat_table in dat_table_list:
        dat_file = dat_file_list[ix]
        write_one_dat_file(dat_table, dat_file)
        ix += 1


def subtest_calc_freq_range():
    hit = {
        "DELTAF": 4.2,
        "DELTAT": 0.42,
        "DriftRate": 0.0,
        'Freq': 4200.0,
        'obs_length': 292.3,
        'max_drift_rate': 4.0
    }
    low1, high1 = calc_freq_range(hit, delta_t=4.2, max_dr=False, follow=True)
    print("subtest_calc_freq_range 1: low={}, high={}".format(low1, high1))
    low2, high2 = calc_freq_range(hit, delta_t=4.2, max_dr=True, follow=False)
    print("subtest_calc_freq_range 2: low={}, high={}".format(low2, high2))
    assert isclose(low1, low2, rtol=RTOL_DIFF)
    assert isclose(high1, high2, rtol=RTOL_DIFF)
    

def test_find_event():
    dat_table_list = [
        DAT1_TABLE,
        DAT2_TABLE,
        DAT3_TABLE,
        DAT4_TABLE,
        DAT5_TABLE               
    ]

    dat_file_list = [
        TESTDIR + "dat1_hits.dat",
        TESTDIR + "dat2_off.dat",
        TESTDIR + "dat3_hits.dat",
        TESTDIR + "dat4_off.dat",
        TESTDIR + "dat5_hits.dat"              
    ]

    write_all_dat_files(dat_table_list, dat_file_list)

    subtest_calc_freq_range()
    
    # Assert that 3 hits are in tables 1, 3, and 5 but only 3 events.
    evt_table = find_events(dat_file_list, SNR_cut=10, check_zero_drift=False, 
                            filter_threshold=3, on_off_first='ON', complex_cadence=False)
    print("evt_table:", evt_table)
    assert len(evt_table) == 3

    # Using the SNR threshold, weed out all but hit 2 in tables 1, 3, and 5 but only 1 event.
    evt_table = find_events(dat_file_list, SNR_cut=200, check_zero_drift=False, 
                            filter_threshold=3, on_off_first='ON', complex_cadence=False)
    print("evt_table:", evt_table)
    assert len(evt_table) == 1

    # Drop filter threshold to 2.  Drop SNR back to 10.
    # Hit 4 in table 1 should be added as a 10th event.
    evt_table = find_events(dat_file_list, SNR_cut=10, check_zero_drift=False, 
                            filter_threshold=2, on_off_first='ON', complex_cadence=False)
    print("evt_table:", evt_table)
    assert len(evt_table) == 10

    # Drop filter threshold to 1.
    # Hit 5 in table 1 should be added as an 11th event.
    evt_table = find_events(dat_file_list, SNR_cut=10, check_zero_drift=False, 
                            filter_threshold=1, on_off_first='ON', complex_cadence=False)
    print("evt_table:", evt_table)
    assert len(evt_table) == 11

    # Keep filter threshold at 1.
    # Add min_drift_rate and max_drift_rate filters.
    # Drop SNR filter.
    evt_table = find_events(dat_file_list, check_zero_drift=False,
                            min_drift_rate=0.1, max_drift_rate=0.4,
                            filter_threshold=1, on_off_first='ON', complex_cadence=False)
    print("evt_table:", evt_table)
    assert len(evt_table) == 10

    # Use on_off_first='OFF'.
    # No events should be found at filter threshold 3.
    evt_table = find_events(dat_file_list, SNR_cut=10, check_zero_drift=False, 
                            filter_threshold=3, on_off_first='OFF', complex_cadence=False)
    print("evt_table:", evt_table)
    assert evt_table == None

    rmtree(TESTDIR, ignore_errors=True)

if __name__ == "__main__":
    test_find_event()
