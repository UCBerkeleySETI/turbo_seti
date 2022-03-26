"""
Read a .dat file.  Use the following filtering parameters to prune it,
saving the original content in .dat.old:
    * min_drift_rate (Hz/s)
    * max_drift_rate (Hz/s)
    * min_snr
"""

import sys
import os
import shutil
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
from pandas.errors import EmptyDataError

DEBUGGING = False
SEP = r"\s+"
HDR_LINE_COUNT = 9
COPIED = False

HELP_EPILOGUE = \
    """
Read a .dat file.  Use the following filtering parameters to prune it,
saving the original content in .dat.original:
    * min_drift_rate (Hz/s)
    * max_drift_rate (Hz/s)
    * min_snr

Exit status:
    0 : All went well, even if 0 top hits were read or retained.
    1 : Some sort of error was reported.
"""


def oops(arg_text):
    print(f"\n*** dat_filter: Oops, {arg_text} !!")
    if COPIED:
        print("*** dat_filter: Note that the original input file has a new name.  Check the dat files carefully !!")
    sys.exit(1)


def main(args=None):
    r"""
    This is the entry point.

    Parameters
    ----------
    args : dict

    """
    # Create an option parser to get command-line input/arguments
    parser = ArgumentParser(description="dat_filter - prune a .dat file.",
                                        formatter_class=RawDescriptionHelpFormatter,
                                        epilog=HELP_EPILOGUE)

    parser.add_argument("dat_file", type=str,
                        help="Path of the .dat file to prune")
    parser.add_argument("-s", "--min_snr", dest="min_snr", type=float, default=None,
                        help="Filter parameter: The SNR below which top hits will be discarded.")
    parser.add_argument("-m", "--min_drift_rate", dest="min_drift_rate", type=float, default=None,
                        help="Filter parameter: TThe drift rate below which top hits will be discarded.")
    parser.add_argument("-M", "--max_drift_rate", dest="max_drift_rate", type=float, default=None,
                        help="Filter parameter: TThe drift rate above which top hits will be discarded.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    full_path = os.path.abspath(args.dat_file)
    if not os.path.exists(full_path):
        oops(f"The .dat file {args.dat_file} does not exist")

    if args.min_drift_rate is None and args.max_drift_rate is None and args.min_snr is None:
        oops("At least one filter parameter must be specified")

    if args.min_drift_rate is not None and args.min_drift_rate < 0:
        oops("The min_drift parameter must be > 0 (absolute value)")
    if args.max_drift_rate is not None and args.max_drift_rate < 0:
        oops("The max_drift parameter must be > 0 (absolute value)")
    if args.min_snr is not None and args.min_snr < 0:
        oops("The min_snr parameter must be > 0 (absolute value)")

    return execute_pruner(full_path, args.min_drift_rate, args.max_drift_rate, args.min_snr)


def execute_pruner(dat_file, min_drift_rate, max_drift_rate, min_snr):
    r"""
    Prune the dat file, governed by the filter parameters.
    Called by main().

    Parameters
    ----------
    dat_file : str
        Path of the .dat file
    min_drift_rate : float
        Minimum allowed drift rate in Hz/s
    max_drift_rate : float
        Maximum allowed drift rate in Hz/s
    min_snr : float
        Minimum allowed SNR
    """

    global COPIED

    # Initisalise.
    tophit_count = 0
    discarded_count = 0
    dat_file_original = dat_file + ".original"

    # Copy the original .dat file.
    try:
        shutil.copyfile(dat_file, dat_file_original)
    except:
        oops(f"Copying {dat_file} to {dat_file_original} failed")
    COPIED = True
    print(f"dat_filter: Copyied {dat_file} to {dat_file_original} - ok")
    print(f"dat_filter: Will now open {dat_file} for output")

    # Input file --> output file.
    with open(dat_file, "w") as out_file:

        # Get all input lines and output the header lines.
        with open(dat_file_original, "r") as in_file:
            for jj in range(HDR_LINE_COUNT):
                try:
                    hdr_line = in_file.readline()
                except:
                    oops("The input file does not appear to be a dat file")
                out_file.write(hdr_line)

        # Make a pandas dataframe out of the tophit lines.
        try:
            df = pd.read_csv(dat_file_original, header=None, sep=SEP, engine="python", comment="#")
        except EmptyDataError:
            # Create empty dataframe.
            df = pd.DataFrame()
        input_count = len(df)

        # Process each tophit.
        for jj in range(input_count):
            drift_rate = abs(float(df[1][jj]))
            snr = abs(float(df[2][jj]))
            if min_drift_rate is not None:
                if abs(drift_rate) < min_drift_rate:
                    discarded_count += 1
                    continue
            if max_drift_rate is not None:
                if abs(drift_rate) > max_drift_rate:
                    discarded_count += 1
                    continue
            if min_snr is not None:
                if abs(snr) < min_snr:
                    discarded_count += 1
                    continue
            # Include this one.
            tophit_count += 1
            tophit_line = '%06d\t'    % (tophit_count)    # Top Hit number
            tophit_line += '%10.6f\t' % df[1][jj]    # Drift Rate
            tophit_line += '%10.6f\t' % df[2][jj]    # SNR
            tophit_line += '%14.6f\t' % df[3][jj]    # Uncorrected Frequency
            tophit_line += '%14.6f\t' % df[4][jj]    # Corrected Frequency
            tophit_line += '%d\t'     % df[5][jj]    # Index
            tophit_line += '%14.6f\t' % df[6][jj]    # freq_start
            tophit_line += '%14.6f\t' % df[7][jj]    # freq_end
            tophit_line += '%s\t'     % df[8][jj]    # SEFD
            tophit_line += '%14.6f\t' % df[9][jj]    # SEFD_mid_freq
            tophit_line += '%i\t'     % df[10][jj]   # fine channel index
            tophit_line += '%i\t'     % df[11][jj]   # total candidates
            out_file.write(tophit_line + "\n")
            if DEBUGGING:
                print(tophit_line)

    print(f"dat_filter: Processed {input_count} top hits.")
    print(f"dat_filter: Retained {tophit_count}.")
    print(f"dat_filter: Discarded {discarded_count}.")
    print(f"dat_filter: Updated file {dat_file}.")
    print(f"dat_filter: The original dat contents are in file {dat_file_original}.")

    return 0


if __name__ == "__main__":
    # Start the show!
    main()
