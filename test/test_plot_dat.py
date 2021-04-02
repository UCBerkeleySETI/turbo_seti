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

with open(H5_LIST_FILE, "w") as f:
    for line in h5_list:
        f.write(line+"\n")

with open(DAT_LIST_FILE, "w") as f:
    for line in dat_list:
        f.write(line+"\n")

dat_table = [[1, -0.363527, 22.32896, 8419.56539, 8419.56539, 651879.0, 8419.567024, 8419.563761, 0.0, 0.0, 0.0, 23111.0],
             [2, -0.35396, 192.893808, 8419.542731, 8419.542731, 659989.0, 8419.544365, 8419.541102, 0.0, 0.0, 0.0, 23111.0],
             [3, -0.363527, 22.572284, 8419.520396, 8419.520396, 667983.0, 8419.52203, 8419.518767, 0.0, 0.0, 0.0, 23111.0]]

HEADER_LINES = [
    "# -------------------------- o --------------------------",
    "# File ID: single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5 ",
    "# -------------------------- o --------------------------",
    "# Source:Voyager1",
    "# MJD: 57650.782094907408	RA: 17h10m03.984s	DEC: 12d10m58.8s",
    "# DELTAT:  18.253611	DELTAF(Hz):  -2.793968",
    "# --------------------------",
    "# Top_Hit_# 	Drift_Rate 	SNR 	Uncorrected_Frequency 	Corrected_Frequency 	Index 	freq_start 	freq_end 	SEFD 	SEFD_freq 	Coarse_Channel_Number 	Full_number_of_hits",
    "# --------------------------"
]

csv_labels = ",TopHitNum,DriftRate,SNR,Freq,ChanIndx,FreqStart,FreqEnd,CoarseChanNum,FullNumHitsInRange,FileID,Source,MJD,RA,DEC,DELTAT,DELTAF,Hit_ID,status,in_n_ons,RFI_in_range,delta_t"

csv_lines = ["0,1,-0.363527,22.32896,8419.56539,651879,8419.567024,8419.563761,0,23111,single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5,VOYAGER-1,59046.926342592589,17h12m40.481s,12d24m13.614s,18.253611,-2.793968,VOYAGER-1_0,on_table_1,2,0,0.0",
             "1,2,-0.35396,192.893808,8419.542731,659989,8419.544365,8419.541102,0,23111,single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5,VOYAGER-1,59046.926342592589,17h12m40.481s,12d24m13.614s,18.253611,-2.793968,VOYAGER-1_1,on_table_1,2,0,0.0"]

CSV = "all_hits.csv"

with open(CSV, "w") as f:
    f.write(csv_labels+"\n")
    for line in csv_lines:
        f.write(line+"\n")

def write_one_dat_file(arg_table, arg_path):
    with open(arg_path, "w") as fh:
        for textline in HEADER_LINES:
            fh.write(textline + "\n")
        for hit_entry in arg_table:
            textline = ""
            for item in hit_entry:
                textline += "{}\t".format(item)
            fh.write(textline + "\n")

write_one_dat_file(dat_table, dat_list[0])

def test_plot_dat():
    OUTDIR = os.getcwd()+'/'
    # will produce one plot
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, CSV, outdir=OUTDIR, 
                        alpha=0.5, window=(8419.542731-793*1e-6, 8419.542731+793e-6))
    
    # will produce no plots
    plot_dat.plot_dat(DAT_LIST_FILE, H5_LIST_FILE, CSV, outdir=OUTDIR, 
                        alpha=0.5, window=(8418.542731-793*1e-6, 8418.542731+793e-6))

if __name__ == "__main__":
    test_plot_dat()