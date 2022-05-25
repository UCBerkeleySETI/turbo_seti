r'''
test_pipelines_4.py

Test plotSETI.
'''

import os
import shutil
from tempfile import gettempdir
import pytest
from turbo_seti import run_pipelines

TESTDIR = gettempdir() + '/pipeline_testing/'
PLOTDIR = TESTDIR + 'plots/'


def execute_one(counter, args):
    print("\n====test_pipelines_4 [{}]================== args: {}".format(counter, args))
    rc = run_pipelines.main(args)
    print("\n====test_pipelines_4 [{}]================== rc: {}".format(counter, rc))
    return rc


@pytest.mark.order(2)
def test_pipelines_4a():
    print('\n===== test_pipelines_4: BEGIN =====')
    
    args = [TESTDIR, "-o", PLOTDIR, "-f", "1", "-s", "25.0", "-c", "on"]
    rc = execute_one(1, args)
    assert(rc == 0)

    args = [TESTDIR, "-o", PLOTDIR, "-f", "2", "-s", "25.0", "-c", "on"]
    rc = execute_one(2, args)
    assert(rc == 0)

    args = [TESTDIR, "-o", PLOTDIR, "-f", "3", "-s", "25.0", "-c", "on"]
    rc = execute_one(3, args)
    assert(rc == 0)

    args = [TESTDIR, "-o", PLOTDIR, "-f", "3", "-m", "0.1", "-c", "on"]
    rc = execute_one(4, args)
    assert(rc == 0)

    args = [TESTDIR, "-o", PLOTDIR, "-f", "2", "-M", "0.4", "-c", "on"]
    rc = execute_one(5, args)
    assert(rc == 0)

    args = [TESTDIR, "-o", PLOTDIR, "-c", "off"]
    rc = execute_one(6, args)
    assert(rc != 0)

    args = [TESTDIR, "-o", PLOTDIR, "-c", "complex", "-n", "Rubbish"]
    rc = execute_one(7, args)
    assert(rc != 0)

    args = [TESTDIR, "-o", PLOTDIR, "-f", "2", "-c", "complex", "-n", "VOYAGER-1"]
    rc = execute_one(8, args)
    assert(rc == 0)

    print('\n===== test_pipelines_4: END =====')

@pytest.mark.order(2)
def test_pipelines_4b():

    # --h5dat_lists
    LISTDIR = TESTDIR + "/my_lists"
    if not os.path.exists(LISTDIR):
        os.mkdir(LISTDIR)
    LISTH5 = "h5_files.lst"
    LISTDAT = "dat_files.lst"
    shutil.copyfile(TESTDIR + "/" + LISTH5, LISTDIR + "/" + LISTH5)
    shutil.copyfile(TESTDIR + "/" + LISTDAT, LISTDIR + "/" + LISTDAT)
    args = [TESTDIR, "-o", PLOTDIR, "-f", "2", "-M", "0.4",
            "--h5dat_lists", LISTDIR + "/" + LISTH5, LISTDIR + "/" + LISTDAT]
    rc = execute_one(42, args)
    assert(rc == 0)


if __name__ == '__main__':
    #test_pipelines_4a()
    test_pipelines_4b()
