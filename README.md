TURBO_SETI
=============================

&nbsp;


## Summary

Based on `dedoppler` [dedoppler](http://github.com/cs150bf/gbt_seti/); which is based on  `rawdopplersearch.c`  [`gbt_seti/src/rawdopplersearch.c`](https://github.com/UCBerkeleySETI/gbt/rawdopplersearch.c))

- Python based, with taylor tree in Cython for improved performance.
- Pre-calculated `drift_indexes`.
- Output text file.

**TO DO**:
This code is under heavy development. I would recommend for now to git pull it every time you use it.

&nbsp;

-------------------

## Installation


### Dependencies

- astropy
- pyximport


&nbsp;

--------------------------

## Usage


### Expected Inputs

At the moment it expects a single .fil file produced by gpuspec.

### Command Line

> **`$ python seti_event.py <FULL_PATH_TO_INPUT_FIL_FILE> [OPTIONS]`**
>
> Use `$ python seti_event.py -h` to view usage details.
>
> &nbsp;
>
> Parameters:
> - `max_drift`:
> - `snr`:
>
> (and so on...)



#### Example:

Currently, there is some voyager test data in bls0 at the GBT cluster.
From the .../turbo_seti/bin/ folder run the next command.

```bash
$ python seti_event.py /datax/eenriquez/voyager_test/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil -p <your_test_folder> -M 2
```

This will take `/datax/eenriquez/voyager_test/test_dedop_bones/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil` as input (and in this particular case it will discover that this file is too big to handle all at once, so it will first partition it into smaller FITS files and save them into the directory specified by option **`-p`**, and then proceed with drift signal search for each small FITS files). Everything else was set to default values.

Known bug:
Once is done creating the FITS files, it will crash.
Just rerun the same command and this time it will understant the FITS files are there and continue with the rest of the analysis.
A bugfix is in progress.


#### Sample Outputs

See `/datax/eenriquez/voyager_test/*/*.log`, `/datax/eenriquez/voyager_test/*.dat` for search results and see `/datax/eenriquez/voyager_test/*.png` for some plots.

&nbsp;

    File ID: blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000_57
    Source:Voyager1 MJD: 57650.782094907408 RA:  17:10:04.0 DEC:  +12:10:58.8       DELTAT:  18.253611      DELTAF(Hz):   2.793968
    --------------------------
    N_candidates: 1055
    --------------------------
    Top Hit #       Drift Rate      SNR     Uncorrected Frequency   Corrected Frequency     Index   freq_start      freq_end        SEFD    SEFD_freq
    --------------------------
    001      -0.353960       51.107710         8419.274366     8419.274366  292536     8419.274344     8419.274386  0.0           0.000000
    002      -0.363527       48.528281         8419.274687     8419.274687  292651     8419.274665     8419.274707  0.0           0.000000
    003      -0.382660      118.779830         8419.297028     8419.297028  300647     8419.297006     8419.297047  0.0           0.000000
    004      -0.392226       51.193226         8419.319366     8419.319366  308642     8419.319343     8419.319385  0.0           0.000000
    005      -0.363527       49.893235         8419.319681     8419.319681  308755     8419.319659     8419.319701  0.0           0.000000
    006       0.000000      298.061948         8419.921871     8419.921871  524287     8419.921848     8419.921890  0.0           0.000000

&nbsp;


### Use as a package

```python
> import dedoppler_bones
```

&nbsp;
--------------------------

