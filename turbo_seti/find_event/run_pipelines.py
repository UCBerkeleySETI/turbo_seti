"""
Main program module for executable plotSETI.
Facilitates the automation of 2 large functions:
    find_event_pipline()
    plot_event_pipline()
"""

import sys
import os
import glob
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import matplotlib

from blimpy import __version__ as BLIMPY_VERSION
from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
from turbo_seti.find_doppler.turbo_seti_version import TURBO_SETI_VERSION

# This file is in the find_event directory.
# The version file is next door in the find_doppler directory (sibling).
CURDIR = os.path.abspath(os.path.join(__file__, os.pardir))
UPDIR = os.path.abspath(os.path.join(CURDIR, os.pardir))
sys.path.append(UPDIR + "/find_doppler")

# 3 standard intermediate files:
NAME_CSVF = "found_event_table.csv"
NAME_H5_LIST = "list_h5_files.txt"
NAME_DAT_LIST = "list_dat_files.txt"

# Error return code
RETURN_NORMAL = 0
RETURN_ERROR = 1

HELP_EPILOGUE = \
"""
Optional Filtering Parameters
--------------------------------------
The following parameters can be used to prune hits from the dat files,
regardless of the filter threshold value:
    * min_drift_rate (Hz/s)
    * max_drift_rate (Hz/s)
    * min_snr

Filter Threshold (ON-OFF tables)
--------------------------------------
1 : Select all top hits from the DAT files.
2 : Select only those top hits that are in at least one ON file AND not in any OFF files.
3 : Select only those top hits that are in all ON files AND not in any OFF files.
Default: 3.

Complex Cadences (--cadence=complex)
------------------------------------
All input .h5/.dat file pairs where the file header source_name fails to match
the --source_name parameter value are bypassed.  In this way, source_name matches are
similar to ON files and non-matches are similar to OFF files.

Using the default --filter_threshold value of 2 means that a top hit must be in
at least one of the matched files to qualify as an event.

Specifying a --filter_threshold value of 3 indicates that a top hit must be in all
matched files to be an event.

Lists of h5 and dat files used internally
-----------------------------------------
Internally, plotSETI uses one text-file-resident list of h5 files
and another for the corresponding dat files.
Normally, these 2 lists are generated internally.
However, if parameter --h5dat_lists is set to 2 file paths separated by spaces
(one text file for h5s, one text file for dats), then those 2 list files:
* Will be checked for existence and consistency.
* Will be used for internal list processing.

E.g. plotSETI --h5dat_lists /dir_a/list_h5_files.txt /b/list_h5_files.txt --out_dir .....
tells plotSETI that there exists a list of h5 files in /dir_a/list_h5_files.txt
and a list of dat files in /dir_b/list_dat_files.txt

If --h5dat_lists is absent (default), plotSETI will internally generate the 2 list files.
"""


def clean_event_stuff(path_out_dir):
    """
    Clean up the output directory of old artifacts.

    Parameters
    ----------
    path_out_dir : str
       Output path of directory holding old artifacts.

    Returns
    -------
    None.

    """
    for deader in glob.glob(f"{path_out_dir}/*.png"):
        os.remove(deader)
    for deader in glob.glob(f"{path_out_dir}/list*.txt"):
        os.remove(deader)
    PATH_CSV = f"{path_out_dir}/{NAME_CSVF}"
    if os.path.exists(PATH_CSV):
        os.remove(PATH_CSV)


def count_text_lines(path_list_file):
    """
    Count the list of text lines in a file.

    Parameters
    ----------
    path_list_file : str
        Path of file containing a list of text lines..

    Returns
    -------
    int
        Count of text lines.

    """
    temp_list = open(path_list_file, "r", encoding="utf-8").readlines()
    return len(temp_list)


def make_lists(path_h5_dir, path_h5_list, path_dat_dir, path_dat_list):
    """
    Create a list of .h5 files and a list of .dat files.

    Parameters
    ----------
    path_h5_dir : str
        Directory where the h5 files reside.
    path_h5_list : str
        Path of output list of h5 files.
    path_dat_dir : str
        Directory where the dat files reside.
    path_dat_list : str
       Path of output list of dat files.

    Returns
    -------
    int
        Number in cadence : Success.
        0                 : Failure.

    """

    N_dat = N_h5 = 0

    print(f"plotSETI: Directory of h5 files: {path_h5_dir}")
    print(f"plotSETI: Directory of dat files: {path_dat_dir}")

    # Make a list of the h5 files.
    with open(path_h5_list, "w", encoding="utf-8") as fh_h5:
        for path_h5 in sorted(glob.glob("{}/*.h5".format(path_h5_dir))):
            N_h5 += 1
            fh_h5.write("{}\n".format(path_h5))
    if N_h5 < 1:
        print("\n*** plotSETI: No h5 files found!")
        return 0
    print(f"plotSETI: Found {N_h5} h5 files.")

    # Make a list of the dat files.
    with open(path_dat_list, "w", encoding="utf-8") as fh_dat:
        for path_dat in sorted(glob.glob("{}/*.dat".format(path_dat_dir))):
            N_dat += 1
            fh_dat.write("{}\n".format(path_dat))
    if N_dat < 1:
        print("\n*** plotSETI: No dat files found!")
        return 0
    print(f"plotSETI: Found {N_dat} dat files.")

    # Make sure that the lists are of the same size.
    if N_h5 != N_dat:
        print("\n*** plotSETI: Count of dat files must = count of h5 files!")
        return 0
    return N_h5


def main(args=None):
    """
    This is the entry point to the plotSETI executable.

    Parameters
    ----------
    args : dict

    """
    # Create an option parser to get command-line input/arguments
    parser = ArgumentParser(description="plotSETI - post-search event-plot utility, version {}."
                                        .format(TURBO_SETI_VERSION),
                                        formatter_class=RawDescriptionHelpFormatter,
                                        epilog=HELP_EPILOGUE)

    parser.add_argument("h5_dir", type=str, default="", nargs="?",
                        help="Path to the directory holding the set of .h5 files")
    parser.add_argument("-d", "--dat_dir", dest="dat_dir", type=str, default=None,
                        help="Path to the directory holding the set of .dat files. Default: h5_path. ")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, default="./",
                        help="Path to the output directory. Default: current directory (.).")
    parser.add_argument("--h5dat_lists", type=str, nargs="+", required=False,
                        help="User-supplied paths to lists of h5 files and dat files.  Default: None supplied; will be internally-generated.")
    parser.add_argument("-f", "--filter_threshold", dest="filter_threshold", type=int,
                        choices=[1, 2, 3], default=3,
                        help="Specification for how strict the top hit filtering will be.")
    parser.add_argument("-z", "--plot_offset", dest="plot_offset", default=False, action="store_true",
                        help="Plot offset lines from from signal?  Default:")
    parser.add_argument("-s", "--snr_threshold", dest="snr_threshold", default=None,
                        help="The SNR below which signals will be discarded.")
    parser.add_argument("-m", "--min_drift_rate", dest="min_drift_rate", default=None,
                        help="The minimum drift rate below which signals will be discarded.")
    parser.add_argument("-M", "--max_drift_rate", dest="max_drift_rate", default=None,
                        help="The maximum drift rate above which signals will be discarded.")
    parser.add_argument("-c", "--cadence", dest="cadence", type=str,
                        choices=["on", "off", "complex"], default="on",
                        help="Input file cadence FIRST file: on source, off source, complex cadence. Default: on.")
    parser.add_argument("-n", "--source_name", dest="source_name", type=str, default="",
                        help="Complex cadence source name.  Don't set this for on or off cadences.")
    parser.add_argument("-e", "--erase_old_files", dest="erase_old", default=True, action="store_true",
                        help="Erase pre-existing *.png, *.csv, and list*.txt files in the output directory.  Default: False.")
    parser.add_argument("-v", "--version", dest="show_version", default=False, action="store_true",
                        help="Show the turbo_seti and blimpy versions and exit.")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Turn on debug tracing (developer).")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.show_version:
        print("turbo_seti: {}".format(TURBO_SETI_VERSION))
        print("blimpy: {}".format(BLIMPY_VERSION))
        return RETURN_NORMAL

    if args.h5_dir == "":
        print("\nThe .h5 directory must be specified!\n")
        os.system("plotSETI -h")
        return RETURN_NORMAL

    if not os.path.exists(args.h5_dir):
        print("\nThe .h5 directory {} does not exist!\n".format(args.h5_dir))
        return RETURN_ERROR

    if args.dat_dir is None:
        args.dat_dir = args.h5_dir

    return execute_pipelines(args)


def execute_pipelines(args):
    """
    Interface to the pipeline functions, called by main().

    Parameters
    ----------
    args : dict
    """

    # Setup some parameter values for find_event_pipeline().
    if args.cadence == "complex":
        complex_cadence = True
        if len(args.source_name) < 1:
            print("\n*** plotSETI: Complex cadence requires a source_name.")
            sys.exit(RETURN_ERROR)

    else:
        complex_cadence = False
        if args.cadence == "on":
            first_file = "ON"
        else:
            first_file = "OFF"
    h5_dir = os.path.abspath(args.h5_dir) + "/"
    dat_dir = os.path.abspath(args.dat_dir) + "/"
    out_dir = os.path.abspath(args.out_dir) + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if args.plot_offset:
        offset="auto"
    else:
        offset=0

    # Establish output pathnames,
    path_csvf = out_dir + NAME_CSVF
    clean_event_stuff(out_dir)

    # Make the h5 and dat lists.
    # Default to auto-generation?
    if args.h5dat_lists is None:
        SZ_user_list = 0
    else:
        SZ_user_list = len(args.h5dat_lists)
    if args.debug:
        print(f"DEBUG h5dats_list: #{SZ_user_list} {args.h5dat_lists}")
    if SZ_user_list == 0: # Default to auto-generation.
        path_h5_list = out_dir + NAME_H5_LIST
        path_dat_list = out_dir + NAME_DAT_LIST
        number_in_cadence = make_lists(h5_dir, path_h5_list, dat_dir, path_dat_list)
        if number_in_cadence == 0:
            return RETURN_ERROR
    else: # User-specified lists
        if SZ_user_list != 2:
            print(f"\n*** plotSETI: h5dat_lists had {SZ_user_list} elements; must be 2 (one for h5 and one for dat)!")
            return RETURN_ERROR
        if args.h5dat_lists[0] is None or args.h5dat_lists[1] is None:
            print(f"\n*** plotSETI: h5dat_lists had {SZ_user_list} elements; must be 2 (one for h5 and one for dat)!")
            return RETURN_ERROR
        # Check the list of h5 files.
        path_h5_list = args.h5dat_lists[0]
        if not os.path.exists(path_h5_list):
            print(f"\n*** plotSETI: File {path_h5_list} does not exist!")
            return RETURN_ERROR
        N_h5 = count_text_lines(path_h5_list)
        print(f"plotSETI: Found {N_h5} h5 files.")
        # Check the list of dat files.
        path_dat_list = args.h5dat_lists[1]
        if not os.path.exists(path_dat_list):
            print(f"\n*** plotSETI: File {path_dat_list} does not exist!")
            return RETURN_ERROR
        N_dat = count_text_lines(path_dat_list)
        print(f"plotSETI: Found {N_dat} dat files.")
        # Make sure that the lists are of the same size.
        if N_h5 != N_dat:
            print("\n*** plotSETI: Count of dat files must = count of h5 files!")
            return RETURN_ERROR
        number_in_cadence = N_h5

    # Run find_event_pipeline()
    if complex_cadence:
        df_check = find_event_pipeline(path_dat_list,
                            path_h5_list,
                            filter_threshold = args.filter_threshold,
                            number_in_cadence = number_in_cadence,
                            on_source_complex_cadence=args.source_name,
                            sortby_tstart=True,
                            check_zero_drift=False,
                            SNR_cut=args.snr_threshold,
                            min_drift_rate=args.min_drift_rate,
                            max_drift_rate=args.max_drift_rate,
                            user_validation=False,
                            csv_name=path_csvf,
                            saving=True)
    else: # not a complex cadence
        df_check = find_event_pipeline(path_dat_list,
                            path_h5_list,
                            filter_threshold = args.filter_threshold,
                            number_in_cadence = number_in_cadence,
                            on_source_complex_cadence=False,
                            on_off_first=first_file,
                            sortby_tstart=True,
                            check_zero_drift=False,
                            SNR_cut=args.snr_threshold,
                            min_drift_rate=args.min_drift_rate,
                            max_drift_rate=args.max_drift_rate,
                            user_validation=False,
                            csv_name=path_csvf,
                            saving=True)

    if df_check is None:
        print("\n*** plotSETI: No events produced in find_event_pipeline()!")
        return RETURN_ERROR

    # Make the plots for all of the HDF5/DAT file pairs in batch mode.
    matplotlib.use("agg", force=True)
    plot_event_pipeline(path_csvf,
                        path_h5_list,
                        plot_dir=out_dir,
                        filter_spec=args.filter_threshold,
                        offset=offset,
                        user_validation=False)

    print(f"\nplotSETI: Plots are stored in directory {out_dir}.")

    return RETURN_NORMAL


if __name__ == "__main__":
    # Start the show!
    main()
