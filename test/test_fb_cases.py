r'''
Package turbo_seti
test/test_fb_cases.py

IMPORTANT:  If the parameters are changed in fb_cases_def.py,
then the following must be executed:  `python3 fb_genref.py`

System concept
--------------
Use setigen to generate small-ish but effective filterbank files.
Then, run turboSETI, producing a DAT file.
Compare resultant DAT file contents to expected results.

Design
------
Entry point for pytest: test_main.
For each maximum drift rate entertained,
    For each test case in the reference (paired records, one each for hit 1 and 2),
        Execute the test case
            1. Generate .fil file with 2 top hits using setigen.
            2. Run turboSETI ---> DAT file.
            3. Get results from DAT file for both hits.
            4. For each hit, compare results to the reference.
'''


import time
import gc
from shutil import rmtree
import numpy as np
from fb_cases_def import THE_MEANING_OF_LIFE, DEBUGGING, TESTDIR, PATH_FIL_FILE, MIN_SNR
from fb_cases_util import generate_fil_file, initialize, make_one_dat_file, \
                            get_case_results, case_comparison


def exec_one_case(case_num, path_fil_file, max_drift, ref_tophit_1, ref_tophit_2):
    r'''Execute one test case'''
    if DEBUGGING:
        print('exec_one_case: on entry, max_drift={}\nref_tophit_1:::{}\nref_tophit_2:::{}'
              .format(max_drift, ref_tophit_1.to_string(), ref_tophit_2.to_string()))
    generate_fil_file(path_fil_file, ref_tophit_1.fdir, ref_tophit_1.drsign)
    make_one_dat_file(path_fil_file, max_drift=max_drift, min_snr=MIN_SNR)
    path_dat_file = path_fil_file.replace('.fil', '.dat')
    obs_tophit_1, obs_tophit_2 = get_case_results(path_dat_file)
    obs_tophit_1.fdir = ref_tophit_1.fdir # replace 0 with correct value
    obs_tophit_1.drsign = ref_tophit_1.drsign # replace 0 with correct value
    obs_tophit_2.fdir = ref_tophit_2.fdir # replace 0 with correct value
    obs_tophit_2.drsign = ref_tophit_2.drsign # replace 0 with correct value
    if DEBUGGING:
        print('exec_one_case: case results\nobs_tophit_1:::{}\nobs_tophit_2:::{}'
              .format(obs_tophit_1.to_string(), obs_tophit_2.to_string()))
    case_comparison(obs_tophit_1, ref_tophit_1, max_drift)
    case_comparison(obs_tophit_2, ref_tophit_2, max_drift)
    gc.collect()
    print('=== CASE {} at max drift {} success'.format(case_num, max_drift))


def run_test_cases(ref_tophit_1, ref_tophit_2, max_drift=None):
    r'''Pytest Entry Point'''
    if max_drift is None:
        raise ValueError('run_test_cases: max_drift not set')
    for jj in range(len(ref_tophit_1)):
        exec_one_case(jj + 1, PATH_FIL_FILE, max_drift, ref_tophit_1[jj], ref_tophit_2[jj])


def test_main(cleanup=True):
    r'''Pytest Entry Point'''
    np.random.seed(THE_MEANING_OF_LIFE) # setigen uses this.
    print('test_main: PATH_FIL_FILE = {}'.format(PATH_FIL_FILE))
    ref_tophit_1, ref_tophit_2 = initialize(TESTDIR)
    run_test_cases(ref_tophit_1, ref_tophit_2, max_drift=5)
    run_test_cases(ref_tophit_1, ref_tophit_2, max_drift=10)
    run_test_cases(ref_tophit_1, ref_tophit_2, max_drift=20)
    run_test_cases(ref_tophit_1, ref_tophit_2, max_drift=30)
    if cleanup:
        rmtree(TESTDIR)


if __name__ == '__main__':
    t1 = time.time()
    test_main(cleanup=False)
    et = (time.time() - t1) / 60.0
    print('test_fb_cases: Elapsed time = {:.2f} min'.format(et))
