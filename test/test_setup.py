r""" Test setup.py"""


def test_setup():
    import os
    cmd = "python3 setup.py check"
    os.system(cmd)
