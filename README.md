[![Build Status](https://travis-ci.org/UCBerkeleySETI/turbo_seti.svg?branch=master)](https://travis-ci.org/UCBerkeleySETI/turbo_seti)
[![Documentation Status](https://readthedocs.org/projects/turbo-seti/badge/?version=latest)](https://turbo-seti.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/UCBerkeleySETI/turbo_seti/branch/master/graph/badge.svg)](https://codecov.io/gh/UCBerkeleySETI/turbo_seti)



TURBO_SETI
=============================

***turbo*SETI** is an analysis tool for the search of narrow band drifting signals in filterbank data (frequency vs. time).
The main purpose of the code is to hopefully one day find signals of extraterrestrial origin!!
It can search the data for hundreds of drift rates (in Hz/sec). It can handle either .fil or .h5 file formats.

**NOTE**:
This code is stable, but new features are currently under development.  'Git pull' for the latest version.

Some details for the expert eye:

- Python based, with taylor tree in Numba for improved performance.
- Pre-calculated `drift index arrays`.
- Output plain text file with information on each hit.
- Including output reader into a pandas DataFrame.

It was originally based on `dedoppler` [dedoppler](http://github.com/cs150bf/gbt_seti/); which is based on  `rawdopplersearch.c`  [`gbt_seti/src/rawdopplersearch.c`](https://github.com/UCBerkeleySETI/gbt_seti/tree/master/src/rawdopplersearch.c))

&nbsp;

-------------------

### Dependencies

- Python 3.7+
- astropy
- numpy
- blimpy (Breakthrough Listen I/O Methods for Python :  https://github.com/UCBerkeleySETI/blimpy)
- pandas
- toolz
- fsspec
- dask
- dask[bag]
- numba
- cupy (NVIDIA GPU mode only)

&nbsp;

--------------------------

## Installation

If you have not yet installed blimpy, do so in this manner:

`python3 -m pip install --user -U git+https://github.com/UCBerkeleySETI/blimpy`

Then, install turbo_seti directly from this repository:

`python3 -m pip install --user -U git+https://github.com/UCBerkeleySETI/turbo_seti`

## NVIDIA GPU Users

Already included is NUMBA Just-in-Time (JIT) CPU performance enhancements. However, if you have NVIDIA GPU hardware on the computer where turbo_seti is going to execute, you can get significant additional performance improvement.  Enable GPU enhanced processing with these steps:

1. Install pypi package "cupy":  `python3 -m pip install cupy`
2. Run the executable this way:  `turboSETI <FULL_PATH_TO_INPUT_HDF5_FILE> -g y [OTHER OPTIONS]`

Once you are using a GPU, you may also want to use the `--single_precision=y` flag, to use float32 rather than float64 processing. Each of
these changes has the potential to roughly double turboSETI's speed, although performance improvements always depend on the details of
your hardware and data set.

## Usage

### Expected Input File Format

At the moment, the `turboSETI` command line and the `FindDoppler` object expect an HDF5 file (.h5) or a Filterbank file (.fil) such as produced by one of the blimpy utilities.

### Usage as a Command Line

Run with data: `turboSETI <FULL_PATH_TO_INPUT_HDF5_FILE> [OPTIONS]`

For an explanation of the program options: `turboSETI -h`



### Usage as a Python Package

```
from turbo_seti.find_doppler.find_doppler import FindDoppler
fdop = FindDoppler(datafile=my_HDF5_file, ...)
fdop.search(...)
```

### Example Usage as a Python Package

```
import time
from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler

H5DIR = "/seti_data/voyager/"
H5PATH = H5DIR + "Voyager1.single_coarse.fine_res.h5"
OUT_DIR_BASE = H5DIR

print("\nUsing HDF5 file: {}\nHeader and data shape:".format(H5PATH))
# -- Get a report of header and data shape
wf = Waterfall(H5PATH)
wf.info()
# -- Instantiate FindDoppler.
print("\nInstantiating the FindDoppler object.")
fdop = FindDoppler(datafile=H5PATH, max_drift=4, snr=25, out_dir=H5DIR)
# -- Search for hits and report elapsed time.
print("\nBegin doppler search.  Please wait ...")
t1 = time.time()
fdop.search()
elapsed_time = time.time() - t1
print("\nFindDoppler.search() elapsed time = {} seconds".format(elapsed_time))
```


### Sample DAT File Output

```
# -------------------------- o --------------------------
# File ID: Voyager1.single_coarse.fine_res.h5 
# -------------------------- o --------------------------
# Source:Voyager1
# MJD: 57650.782094907408	RA: 17h10m03.984s	DEC: 12d10m58.8s
# DELTAT:  18.253611	DELTAF(Hz):  -2.793968
# --------------------------
# Top_Hit_# 	Drift_Rate 	SNR 	Uncorrected_Frequency 	Corrected_Frequency 	Index 	freq_start 	freq_end 	SEFD 	SEFD_freq 	Coarse_Channel_Number 	Full_number_of_hits 	
# --------------------------
001	 -0.392226	 30.612128	   8419.319368	   8419.319368	739933	   8419.321003	   8419.317740	0.0	      0.000000	0	858	
002	 -0.373093	245.707984	   8419.297028	   8419.297028	747929	   8419.298662	   8419.295399	0.0	      0.000000	0	858	
003	 -0.392226	 31.220652	   8419.274374	   8419.274374	756037	   8419.276009	   8419.272745	0.0	      0.000000	0	858	
```


### Sample Console Logging (level=INFO) Output
Note that the coarse channel number appears as a suffix of the logger name.  For example, "find_doppler.8" depicts logging for find_doppler.py in coarse channel number 8 (relative to 0).
```
Using HDF5 file: /seti_data/voyager/Voyager1.single_coarse.fine_res.h5
Header and data shape:

--- File Info ---
DIMENSION_LABELS :   ['frequency' 'feed_id' 'time']
        az_start :                              0.0
       data_type :                                1
            fch1 :            8421.386717353016 MHz
            foff :      -2.7939677238464355e-06 MHz
           ibeam :                                1
      machine_id :                               20
          nbeams :                                1
           nbits :                               32
          nchans :                          1048576
            nifs :                                1
     rawdatafile : guppi_57650_67573_Voyager1_0002.0000.raw
     source_name :                         Voyager1
         src_dej :                       12:10:58.8
         src_raj :                     17:10:03.984
    telescope_id :                                6
           tsamp :                     18.253611008
   tstart (ISOT) :          2016-09-19T18:46:13.000
    tstart (MJD) :                57650.78209490741
        za_start :                              0.0

Num ints in file :                               16
      File shape :                 (16, 1, 1048576)
--- Selection Info ---
Data selection shape :                 (16, 1, 1048576)
Minimum freq (MHz) :                8418.457032646984
Maximum freq (MHz) :                8421.386717353016

Instantiating the FindDoppler object.
find_doppler.0  INFO     {'DIMENSION_LABELS': array(['frequency', 'feed_id', 'time'], dtype=object), 'az_start': 0.0, 'data_type': 1, 'fch1': 8421.386717353016, 'foff': -2.7939677238464355e-06, 'ibeam': 1, 'machine_id': 20, 'nbeams': 1, 'nbits': 32, 'nchans': 1048576, 'nifs': 1, 'rawdatafile': 'guppi_57650_67573_Voyager1_0002.0000.raw', 'source_name': 'Voyager1', 'src_dej': <Angle 12.183 deg>, 'src_raj': <Angle 17.16777333 hourangle>, 'telescope_id': 6, 'tsamp': 18.253611008, 'tstart': 57650.78209490741, 'za_start': 0.0}

Begin doppler search.  Please wait ...
find_doppler.0  INFO     File: /seti_data/voyager/Voyager1.single_coarse.fine_res.h5
 drift rates (min, max): (0.000000, 4.000000)
 SNR: 25.000000

Starting ET search using /seti_data/voyager/Voyager1.single_coarse.fine_res.h5
find_doppler.0  INFO     Parameters: datafile=/seti_data/voyager/Voyager1.single_coarse.fine_res.h5, max_drift=4, min_drift=0.0, snr=25, out_dir=/seti_data/voyager/, coarse_chans=None, flagging=False, n_coarse_chan=None, kernels=None, gpu_backend=False, precision=2, append_output=False, log_level_int=20, obs_info={'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0, 'pulsar_stats': array([0., 0., 0., 0., 0., 0.]), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0, 'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}
find_doppler.0  INFO     Top hit found! SNR 30.612128, Drift Rate -0.392226, index 739933
find_doppler.0  INFO     Top hit found! SNR 245.707984, Drift Rate -0.373093, index 747929
find_doppler.0  INFO     Top hit found! SNR 31.220652, Drift Rate -0.392226, index 756037

FindDoppler.search() elapsed time = 9.972093105316162 seconds
```

&nbsp;
--------------------------
