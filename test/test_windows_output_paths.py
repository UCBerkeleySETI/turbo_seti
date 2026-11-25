r"""
Regression tests for Windows output paths: .dat/.log paths were built by splitting the input
path on '/' and joining with '/', so on Windows they became e.g.
C:\out/C:\data\file.dat  (invalid) instead of  C:\out\file.dat.
The checks use ntpath / posixpath, so they run on any OS (including Linux CI).
"""
import ntpath
import posixpath

import pytest

from turbo_seti.find_doppler.find_doppler import output_path


@pytest.mark.parametrize("out_dir", [r"C:\Users\me\out", "C:\\Users\\me\\out\\"])
def test_output_path_windows(out_dir):
    got = output_path(out_dir, r"C:\Users\me\data\Parkes_57767_56757_HIP33690_S_fine.h5", ".dat", path=ntpath)
    assert got == r"C:\Users\me\out\Parkes_57767_56757_HIP33690_S_fine.dat"


def test_output_path_windows_coarse_channel_suffix():
    got = output_path(r"D:\o", r"D:\d\voyager.fil", "_3.log", path=ntpath)
    assert got == r"D:\o\voyager_3.log"


@pytest.mark.parametrize("out_dir", ["/tmp/out", "/tmp/out/"])
def test_output_path_posix_unchanged(out_dir):
    # same result as the previous '/'.split/join code on POSIX
    got = output_path(out_dir, "/data/voyager.h5", ".dat", path=posixpath)
    assert got == "/tmp/out/voyager.dat"
