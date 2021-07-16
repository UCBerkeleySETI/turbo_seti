#!/usr/bin/env python
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

# This file is in the find_event directory.
# The version file is next door in the find_doppler directory (sibling).
CURDIR = os.path.abspath(os.path.join(__file__, os.pardir))
UPDIR = os.path.abspath(os.path.join(CURDIR, os.pardir))
sys.path.append(UPDIR + "/find_doppler")
from turbo_seti_version import TURBO_SETI_VERSION

# 3 standard intermediate files:
NAME_CSVF = "found_event_table.csv"
NAME_H5_LIST = "list_h5_files.txt"
NAME_DAT_LIST = "list_dat_files.txt"


HELP_EPILOGUE = \
"""
Top Hit Filtering (--filter_threshold)
--------------------------------------
1 : Select all top hits from the DAT file as events.
2 : Select only those top hits that are in at least one ON file AND not in any OFF files.
3 : Select only those top hits that are in all ON files AND not in any OFF files.
Default: 2.

Complex Cadences (--cadence=complex)
------------------------------------
All input .h5/.dat file pairs where the file header source_name fails to match
the --source_name parameter value are bypassed.  In this way, source_name matches are
similar to ON files and non-matches are similar to OFF files.

Using the default --filter_threshold value of 2 means that a top hit must be in
at least one of the matched files to qualify as an event.

Specifying a --filter_threshold value of 3 indicates that a top hit must be in all
matched files to be an event.
"""


def clean_event_stuff(path_out_dir):
    r"""Take out the trash."""
    for deader in glob.glob("{}/*.csv".format(path_out_dir)):
        os.remove(deader)
    for deader in glob.glob("{}/*.png".format(path_out_dir)):
        os.remove(deader)
    for deader in glob.glob("{}/list*.tst".format(path_out_dir)):
        os.remove(deader)


def make_lists(path_h5_dir, path_h5_list, path_dat_list):
    r"""Create 2 files, a list of .h5 files and a list of .dat files."""
    with open(path_dat_list, "w") as fh_dat:
        with open(path_h5_list, "w") as fh_h5:
            for path_h5 in sorted(glob.glob("{}/*.h5".format(path_h5_dir))):
                path_dat = path_h5.replace(".h5", ".dat")
                fh_h5.write("{}\n".format(path_h5))
                fh_dat.write("{}\n".format(path_dat))


def main(args=None):
    r"""
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

    parser.add_argument("input_dir_path", type=str, default="", nargs="?",
                        help="Path to the directory holding the set of .h5/.dat file pairs")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, default="./",
                        help="Path to the output directory. Default: current directory (.). ")
    parser.add_argument("-f", "--filter_threshold", dest="filt_thresh", type=int,
                        choices=[1, 2, 3], default=2,
                        help="Specification for how strict the top hit filtering will be.")
    parser.add_argument("-s", "--snr_threshold", dest="snr_threshold", type=float, default=10.0,
                        help="The SNR below which signals will be discarded.")
    parser.add_argument("-c", "--cadence", dest="cadence", type=str,
                        choices=["on", "off", "complex"], default="on",
                        help="Input file cadence FIRST file: on source, off source, complex cadence. Default: on.")
    parser.add_argument("-n", "--source_name", dest="source_name", type=str, default="",
                        help="Complex cadence source name.  Don't set this for on or off cadences.")
    parser.add_argument("-e", "--erase_old_files", dest="erase_old", default=True, action="store_true",
                        help="Erase pre-existing *.png, *.csv, and list*.txt files in the output directory.  Default: False.")
    parser.add_argument("-v", "--version", dest="show_version", default=False, action="store_true",
                        help="Show the turbo_seti and blimpy versions and exit.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.show_version:
        print("turbo_seti: {}".format(TURBO_SETI_VERSION))
        print("blimpy: {}".format(BLIMPY_VERSION))
        return 0

    if args.input_dir_path == "":
        os.system("plotSETI -h")
        return 0

    if not os.path.exists(args.input_dir_path):
        print("\nInput directory {} does not exist!\n".format(args.input_dir_path))
        return 86

    return execute_pipelines(args)


def execute_pipelines(args):
    r"""
    Interface to the pipeline functions, called by main().

    Parameters
    ----------
    args : dict
    """

    # Setup some parameter values for find_event_pipeline().
    if args.cadence == "complex":
        complex_cadence = True
        if len(args.source_name) < 1:
            print("\n*** plotSETI: Complex cadence requires a source_name.  Bye-bye.")
            sys.exit(86)
            
    else:
        complex_cadence = False
        if args.cadence == "on":
            first_file = "ON"
        else:
            first_file = "OFF"
    from_dir = os.path.abspath(args.input_dir_path) + "/"
    dest_dir = os.path.abspath(args.out_dir) + "/"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # Establish output pathnames,
    path_h5_list = dest_dir + NAME_H5_LIST
    path_dat_list = dest_dir + NAME_DAT_LIST
    path_csvf = dest_dir + NAME_CSVF
    clean_event_stuff(dest_dir)
    make_lists(from_dir, path_h5_list, path_dat_list)

    # Run find_event_pipeline()
    number_in_cadence = len(open(path_h5_list).readlines())
    if complex_cadence:
        df_check = find_event_pipeline(path_dat_list,
                            filter_threshold = args.filt_thresh,
                            number_in_cadence = number_in_cadence,
                            on_source_complex_cadence=args.source_name,
                            sortby_tstart=True,
                            check_zero_drift=False,
                            SNR_cut=args.snr_threshold,
                            user_validation=False,
                            csv_name=path_csvf,
                            saving=True)
    else: # not a complex cadence
        df_check = find_event_pipeline(path_dat_list,
                            filter_threshold = args.filt_thresh,
                            number_in_cadence = number_in_cadence,
                            on_source_complex_cadence=False,
                            on_off_first=first_file,
                            sortby_tstart=True,
                            check_zero_drift=False,
                            SNR_cut=args.snr_threshold,
                            user_validation=False,
                            csv_name=path_csvf,
                            saving=True)

    if df_check is None:
        print("\n*** plotSETI: No events produced in find_event_pipeline()!  Bye-bye.")
        return 86

    # Make the plots for all of the HDF5/DAT file pairs in batch mode.
    matplotlib.use("agg", force=True)
    plot_event_pipeline(path_csvf,
                        path_h5_list,
                        plot_dir=dest_dir,
                        filter_spec=args.filt_thresh,
                        offset=0,
                        user_validation=False)

    print("\nplotSETI: Plots are stored in directory {}.  Bye-bye.".format(dest_dir))
    
    return 0


if __name__ == "__main__":
    # Start the show!
    main()

