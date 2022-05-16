"""
dat_diff - show differences between two dat files.

Two processes in succession:
1) For each entry in dat file dat1, look for a match in dat file dat2.
2) For each entry in dat file dat2, look for a match in dat file dat1.

The comparison between dataframe entries uses the following dataframe columns:
* Coarse channel number (exact match)
* Frequency (within rtol) - see numpy.isclose() or math.isclose() for definition.
* Drift rate (within rtol)
"""

import os
import sys

from argparse import ArgumentParser
import pandas as pd
import numpy as np

DEBUGGING = False
SEP = r'\s+'


def oops(msg):
    """
    Display an error message and exit to the O/S.

    Parameters
    ----------
    msg : str
        Error message for display.

    Returns
    -------
    None.

    """
    print(f"\n*** Oops, {msg} !!")
    sys.exit(86)


def finder(df1, path1, df2, path2, etol):
    """
    For each entry in data frame df1, try to find a matching entry in data frame df2.

    Parameters
    ----------
    df1 : Pandas DataFrame
        Reference dataframe.
    path1 : str
        Path of external file where df1 was sourced from.
    df2 : str
        Dataframe to be searched for a match.
    path2 : str
        Path of external file where df2 was sourced from..
    etol : float
        Relative error tolerance.  See RTOL parameter in numpy.isclose() or math.isclose().

    Returns
    -------
    None.

    """

    # Extract file names.
    file1 = os.path.basename(path1)
    file2 = os.path.basename(path2)
    print(f"\n===== For each record in {file1}, try to find a corresponding one in {file2}.")
    ndf1 = len(df1)
    ndf2 = len(df2)
    print(f"===== {ndf1} records in {file1}, {ndf2} records in {file2}.")

    # Main loop.
    notfound_count = 0
    for ii in range(ndf1):
        if DEBUGGING:
            print(f"DEBUG ii={ii}")
        drate1 = df1[1][ii]
        snr1 = df1[2][ii]
        freq1 = df1[3][ii]
        cchan1 = df1[10][ii]
        flag_match = False
        for jj in range(ndf2):
            if DEBUGGING:
                print(f"DEBUG ii={ii}, jj={jj}")
            drate2 = df2[1][jj]
            #snr2 = df2[2][jj]
            freq2 = df2[3][jj]
            cchan2 = df2[10][jj]
            if cchan1 != cchan2:
                continue
            if not np.isclose(drate1, drate2, rtol=etol):
                continue
            if not np.isclose(freq1, freq2, rtol=etol):
                continue
            flag_match = True
            break
        if not flag_match:
            notfound_count += 1
            print(f"{file1} {ii} not found in {file2} >  Coarse  {cchan1}  drate  {drate1}  Freq  {freq1}  SNR  {snr1}")

    # The End.
    if notfound_count == 0:
        print(f"{file1} : {file2} > All found.")
    else:
        print(f"{file1} : {file2} > Not found count = {notfound_count}")


def main(args=None):
    """
    Executable program entry point.
    If args=None, then the command-line holds the list of arguments;
    Else, args holds the list of arguments.

    Parameters
    ----------
    args : list, optional
        List of arguments. The default is None.

    Returns
    -------
    None.

    """

    # Parse and validate arguments.
    parser = ArgumentParser(description="Find differences between 2 DAT files.")
    parser.add_argument("dat1", type=str,
                        help="Path of the 1st DAT file.")
    parser.add_argument("dat2", type=str,
                        help="Path of the 2nd DAT file.")
    parser.add_argument("-e", "--etol", dest="etol", type=float, default=0.0001,
                        help="Relative error tolerance (see numpy.isclose() or math.isclose(). Default=0.0001 (0.01%%).")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    print(f"Relative error tolerance = {args.etol}")

    # Make sure that the DAT path #1 exists.
    # Load the data frames.
    if not os.path.exists(args.dat1):
        oops(f"DAT file {args.dat1} does not exist")
    df1 = pd.read_csv(args.dat1, header=None, sep=SEP, engine="python", comment="#")
    nrows1 = len(df1)
    if nrows1 < 1:
        oops(f"DAT file {args.dat1} is empty")

    # Make sure that the DAT path #2 exists.
    # Load the data frames.
    if not os.path.exists(args.dat2):
        oops(f"DAT file {args.dat2} does not exist")
    df2 = pd.read_csv(args.dat2, header=None, sep=SEP, engine="python", comment="#")
    nrows2 = len(df2)
    if nrows2 < 1:
        oops(f"DAT file {args.dat2} is empty")

    finder(df1, args.dat1, df2, args.dat2, args.etol)
    finder(df2, args.dat2, df1, args.dat1, args.etol)


if __name__ == "__main__":
    main()
