r'''test_plot_dat - test plotting of a DAT file.
NOTE:  This source file uses data downloaded by test_pipelines_1.py
'''

import glob
from tempfile import gettempdir
from turbo_seti.find_event import plot_dat

TESTDIR = gettempdir() + '/pipeline_testing/'
PLOTDIR = TESTDIR + 'plots/'
h5_list = sorted(glob.glob(TESTDIR + 'single*.h5'))
dat_list = sorted(glob.glob(TESTDIR + 'single*.dat'))
H5_LIST_FILE = "list_h5_files.txt"
DAT_LIST_FILE= "list_dat_files.txt"
PATH_CSVF = TESTDIR + 'found_event_table.csv'

with open(H5_LIST_FILE, "w") as f:
    for line in h5_list:
        f.write(line + "\n")
with open(DAT_LIST_FILE, "w") as f:
    for line in dat_list:
        f.write(line + "\n")

def test_plot_dat():
    # test default settings, will produce six plots, all candidates
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, PATH_CSVF)

    # will produce no plots
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, PATH_CSVF, outdir=PLOTDIR, 
                        alpha=0.5, window=(8418.542731-793e-6, 8418.542731+793e-6))

    # will produce three candidate plots
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, PATH_CSVF, outdir=PLOTDIR, 
                        alpha=0.5, window=(8419.542731-793e-6, 8419.542731+793e-6))

    # will produce one plot, no candidates
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, PATH_CSVF, outdir=PLOTDIR, 
                        window=(8419.519896-2.5e-3, 8419.519896+2.5e-3))

if __name__ == "__main__":
    test_plot_dat()