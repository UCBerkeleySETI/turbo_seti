r'''
TESTDAT_filter.py

Test dat_filter.
'''

from tempfile import gettempdir
import shutil
import pandas as pd
from pandas.errors import EmptyDataError
from turbo_seti.find_event import dat_filter

TESTDIR = gettempdir() + '/pipeline_testing/'
TESTDAT = TESTDIR + "TESTDAT_filter.dat"
SEP = r"\s+"


def execute_one(counter, args):
    print("\n====TESTDAT_filter [{}]================== args: {}".format(counter, args))

    shutil.copyfile(TESTDIR + "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.dat", TESTDAT)
    rc = dat_filter.main(args)
    print("\n====TESTDAT_filter [{}]================== rc: {}".format(counter, rc))
    assert rc == 0
    try:
        df = pd.read_csv(TESTDAT, header=None, sep=SEP, engine="python", comment="#")
    except EmptyDataError:
        # Create empty dataframe.
        df = pd.DataFrame()
    return len(df)


def test_dat_filter():
    print('\n===== TESTDAT_filter: BEGIN =====')

    args = ["-s", "10", "-m", "0.1", "-M", "0.4", TESTDAT]
    len_df = execute_one(1, args)
    assert len_df == 3

    args = ["-m", "0.36", TESTDAT]
    len_df = execute_one(2, args)
    assert len_df == 2

    args = ["-M", "0.34", TESTDAT]
    len_df = execute_one(3, args)
    assert len_df == 0

    args = ["-s", "100", TESTDAT]
    len_df = execute_one(3, args)
    assert len_df == 1

    print('\n===== TESTDAT_filter: END =====')


if __name__ == '__main__':
    test_dat_filter()
