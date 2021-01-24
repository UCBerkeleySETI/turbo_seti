r'''
Package turbo_seti
test/fb_genref.py

Generate a reference file for subsequent use by test_fb_cases.py.
'''

from os import mkdir
from shutil import rmtree
import time
import numpy as np
from fb_cases_def import THE_MEANING_OF_LIFE, TESTDIR, PATH_FIL_FILE, MIN_SNR
from fb_cases_util import generate_fil_file, make_one_dat_file, get_case_results

PATH_REF = TESTDIR + 'fb_dat_reference.txt'
MAX_DRIFT = 5


def add_one(arg_case_num, arg_fh, arg_fdir, arg_drsign):
    r'''Add one case to the reference file'''
    print('fb_genref: Case {} ...'.format(arg_case_num))
    generate_fil_file(PATH_FIL_FILE, arg_fdir, arg_drsign)
    make_one_dat_file(PATH_FIL_FILE, max_drift=MAX_DRIFT, min_snr=MIN_SNR)
    path_dat_file = PATH_FIL_FILE.replace('.fil', '.dat')
    obs_tophit_1, obs_tophit_2 = get_case_results(path_dat_file)
    arg_fh.write('#\n')
    arg_fh.write('# Case {}: frequency {}, drift rate {}\n'
                 .format(arg_case_num, arg_fdir, arg_drsign))
    FMT_RECORD = '{}       {}     {}          {}        {}      {}    {}    {}\n'
    
    record_1 = FMT_RECORD \
        .format(arg_case_num, arg_fdir, arg_drsign, obs_tophit_1.tophit_id, obs_tophit_1.drate,
                obs_tophit_1.snr, obs_tophit_1.freq, obs_tophit_1.index)
    arg_fh.write(record_1)

    record_2 = FMT_RECORD \
        .format(arg_case_num, arg_fdir, arg_drsign, obs_tophit_2.tophit_id, obs_tophit_2.drate,
                obs_tophit_2.snr, obs_tophit_2.freq, obs_tophit_2.index)
    arg_fh.write(record_2)


np.random.seed(THE_MEANING_OF_LIFE) # setigen uses this.
rmtree(TESTDIR, ignore_errors=True)
mkdir(TESTDIR)
t1 = time.time()
print('fb_genref: Begin generating {}'.format(PATH_REF))

with open(PATH_REF, 'w') as file_handle:
    file_handle.write('CaseNr  fdir  drsign     tophit   drate     snr            freq         index\n')
    file_handle.write('#=====  ====  ======     ======   =====     ===            ====         =====\n')
    add_one(1, file_handle, +1, +1)
    add_one(2, file_handle, +1, -1)
    add_one(3, file_handle, -1, +1)
    add_one(4, file_handle, -1, -1)
    file_handle.write('#\n')
    file_handle.write('#---------END---------------------------------------\n')
    file_handle.close()

et = (time.time() - t1) / 60.0
print('fb_genref: Elapsed time = {:.2f} min'.format(et))
