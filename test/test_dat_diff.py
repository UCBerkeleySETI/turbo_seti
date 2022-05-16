""" Test find_doppler/dat_diff.py """
import os
import pytest
from turbo_seti.find_doppler.dat_diff import main

TESTDIR = os.path.split(os.path.abspath(__file__))[0]
VOYADAT = os.path.join(TESTDIR, "Voyager1.single_coarse.fine_res.dat")
VOYADATFLIPPED = os.path.join(TESTDIR, "Voyager1.single_coarse.fine_res.flipped.dat")


@pytest.mark.order(index=-3)
def test_dat_diff_help(capsys):

    with pytest.raises(SystemExit) as exit_code:
        args = ["-h"]
        main(args)
    out, err = capsys.readouterr()
    print(out, err)
    assert exit_code.type == SystemExit
    assert exit_code.value.code == 0


@pytest.mark.order(index=-2)
def test_dat_diff_2_dats():

    args = [VOYADAT, VOYADATFLIPPED]
    main(args)


@pytest.mark.order(index=-1)
def test_dat_diff_missing_dat(capsys):

    with pytest.raises(SystemExit) as exit_code:
        args = ["nonexistent.dat", VOYADATFLIPPED]
        main(args)
    out, err = capsys.readouterr()
    print(out, err)
    assert exit_code.type == SystemExit
    assert exit_code.value.code != 0
