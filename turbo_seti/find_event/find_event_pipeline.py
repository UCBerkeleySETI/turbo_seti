#!/usr/bin/env python

r"""
Front-facing script to find drifting, narrowband events in a set of generalized
cadences of ON-OFF radio SETI observations.

The main function contained in this file is :func:`find_event_pipeline` calls
find_events from find_events.py to read a list of turboSETI .dat files.
It then finds events within this group of files.
"""

#required packages and programs
import os
from operator import attrgetter

import logging
logger_name = 'find_event_pipeline'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np
from blimpy import Waterfall
from blimpy.utils import change_the_ext
from turbo_seti.find_event.find_event import find_events


RTOL_DIFF = 0.01 # 1%


class PathRecord:
    r''' Definition of a DAT record '''
    def __init__(self, path_dat, tstart, source_name, fch1, foff, nchans):
        self.path_dat = path_dat
        self.tstart = tstart
        self.source_name = source_name
        self.fch1 = fch1
        self.foff = foff
        self.nchans = nchans
    def __repr__(self):
        return repr((self.path_dat, self.tstart, self.source_name))


def get_file_header(filepath_h5):
    r'''
    Extract and return the target's source name from the DAT file path.

    Parameters
    ----------
    dat_path : str
        Full or relative path name of the DAT file

    Returns
    -------
    header : Waterfall header object



    '''
    wf = Waterfall(filepath_h5, load_data=False)
    return wf.container.header


def close_enough(x, y):
    r"""Make sure that x and y are close enough to be considered roughly equal."""
    if np.isclose(float(x), float(y), rtol=RTOL_DIFF):
        return True
    return False


def find_event_pipeline(dat_file_list_str,h5_file_list_str=None, check_zero_drift=False, filter_threshold=3,
                        on_off_first='ON', number_in_cadence=6, on_source_complex_cadence=False,
                        saving=True, csv_name=None, user_validation=False,
                        sortby_tstart=True,
                        SNR_cut=None, min_drift_rate=None, max_drift_rate=None):
    """
    Find event pipeline.

    Parameters
    ----------
    dat_file_list_str : str
        The string name of a plaintext file ending in .lst
        that contains the filenames of .dat files, each on a
        new line, that were created with seti_event.py. The
        .lst should contain a set of cadences (ON observations
        alternating with OFF observations). The cadence can be
        of any length, given that the ON source is every other
        file. This includes Breakthrough Listen standard ABACAD
        as well as OFF first cadences like BACADA. Minimum
        cadence length is 2, maximum cadence length is
        unspecified (currently tested up to 6).
        Example: ABACAD|ABACAD|ABACAD
    h5_file_list_str : str | None
        The string name of a plaintext file ending in .lst
        that contains the filenames of .h5 files, each on a
        new line, that were created with seti_event.py. The
        .lst should contain a set of cadences (ON observations
        alternating with OFF observations). The cadence can be
        of any length, given that the ON source is every other
        file. This includes Breakthrough Listen standard ABACAD
        as well as OFF first cadences like BACADA. Minimum
        cadence length is 2, maximum cadence length is
        unspecified (currently tested up to 6).
    check_zero_drift : bool
        A True/False flag that tells the program whether to
        include hits that have a drift rate of 0 Hz/s. Earth-
        based RFI tends to have no drift rate, while signals
        from the sky are expected to have non-zero drift rates.
    filter_threshold : int, default is 3
        Specification for how strict the hit filtering will be.
        There are 3 different levels of filtering, specified by
        the integers 1, 2, and 3.
        * Filter_threshold = 1 applies the following parameter checks:
            check_zero_drift
            SNR_cut
            min_drift_rate
            max_drift_rate
        However, Filter_threshold = 1 applies no ON-OFF check.
        * Filter_threshold = 2 returns hits that passed level 1
        AND that are in at least one ON table but no OFF tables.
        * Filter_threshold = 3 returns events that passed level 2
        AND that are present in *ALL* ON tables.
    on_off_first : str {'ON', 'OFF'}
        Tells the code whether the .dat sequence starts with
        the ON or the OFF observation. Valid entries are 'ON'
        and 'OFF' only. Default is 'ON'.
    number_in_cadence : int
        The number of files in a single ON-OFF cadence.
        Default is 6 for ABACAD.
    on_source_complex_cadence : bool
        If using a complex cadence (i.e. ons and offs not
        alternating), this variable should be the string
        target name used in the .dat filenames. The code will
        then determine which files in your dat_file_list_str
        cadence are ons and which are offs.
    saving : bool
        A True/False flag that tells the program whether to
        save the output array as a .csv.
    user_validation : bool
        A True/False flag that, when set to True, asks if the
        user wishes to continue with their input parameters
        (and requires a 'y' or 'n' typed as confirmation)
        before beginning to run the program. Recommended when
        first learning the program, not recommended for
        automated scripts.
    sortby_tstart : bool
        If True, the input file list is sorted by header.tstart.
    SNR_cut : None (default value) or float value > 0
        If None, then all SNR values from the dedoppler results in the dat
        files are accepted as-is.
        Otherwise, the specified value is the threshold SNR below which
        hits will be discarded.
    min_drift_rate : None (default value) or float value > 0
        If None, then all drift rate values from the dedoppler results in the dat
        files are accepted as-is.
        Otherwise, the specified value is the threshold drift rate below which
        hits will be discarded.
    max_drift_rate : None (default value) or float value > 0
        If None, then all drift rate values from the dedoppler results in the dat
        files are accepted as-is.
        Otherwise, the specified value is the threshold drift rate above which
        hits will be discarded.

    Returns
    -------
    Either:
        * a Pandas dataframe with all the events that were found.
        * None, if no events were found.

    Notes
    -----
    The HDF5 file is ASSUMED(!!) to have the same name as .dat files.

    Examples
    --------
    >>> import find_event_pipeline;
    >>> find_event_pipeline.find_event_pipeline(dat_file_list_str,
    ...                                         SNR_cut=10,
    ...                                         min_drift_rate=0.1,
    ...                                         max_drift_rate=4,
    ...                                         check_zero_drift=False,
    ...                                         filter_threshold=3,
    ...                                         on_off_first='ON',
    ...                                         number_in_cadence=6,
    ...                                         on_source_complex_cadence=False,
    ...                                         saving=True,
    ...                                         user_validation=False)

    """
    print()
    print("===========   BEGINNING FIND_EVENT PIPELINE   ===========")
    print()

    if on_source_complex_cadence:
        print("Assuming a complex cadence for the following on source: {}"
              .format(on_source_complex_cadence))
    else: # not on_source_complex_cadence:
        print("Assuming the first observation is an " + on_off_first)
        complex_cadence = on_source_complex_cadence

    # Get a list of the DAT files.
    # Get source names and build path_record list.
    source_name_list = []
    path_record = []

    # Get a list of the DAT/h5 files.
    def list_of_files(dat_file_list_str):
        dat_file_list = open(dat_file_list_str).readlines()
        dat_file_list = [files.replace('\n','') for files in dat_file_list]
        dat_file_list = [files.replace(',','') for files in dat_file_list]
        n_files = len(dat_file_list)
        return n_files, dat_file_list
    n_files, dat_file_list = list_of_files(dat_file_list_str)
    if h5_file_list_str is None:
        h5_file_list = dat_file_list
        for hf in h5_file_list:
            header = get_file_header(change_the_ext(hf, 'dat', 'h5'))
            source_name = header["source_name"]
            tstart = header["tstart"]
            path_record.append(PathRecord(hf, tstart, source_name, header["fch1"],
                                        header["foff"], header["nchans"]))
            source_name_list.append(source_name)
    else:
        _, h5_file_list = list_of_files(h5_file_list_str)
        for hf in h5_file_list:
            header = get_file_header(hf)
            for dat in dat_file_list: # O(n^2) TODO: create tests in pytest
                if os.path.basename(dat).replace('.dat','.h5')==os.path.basename(hf):
                    source_name = header["source_name"]
                    tstart = header["tstart"]
                    path_record.append(PathRecord(dat, tstart, source_name, header["fch1"],
                                            header["foff"], header["nchans"]))
                    source_name_list.append(source_name)



    # If sorting by header.tstart, then rewrite the dat_file_list in header.tstart order.
    if sortby_tstart:
        path_record = sorted(path_record, key=attrgetter('tstart'))
        dat_file_list = []
        for obj in path_record:
            dat_file_list.append(obj.path_dat)

    # Set up the frequency range matcher record.
    # If a complex cadence, the source name is used to select the matcher;
    # Otherwise, just use the first record.
    if on_source_complex_cadence:
        flag_terminate = True
        for obj in path_record: # Look for 1st occurence of source_name.
            if obj.source_name == on_source_complex_cadence:
                matcher = obj
                flag_terminate = False
                break
        if flag_terminate:
            logger.error(f"Source '{on_source_complex_cadence}' is not in this complex cadence!")
            for obj in path_record:
                logger.info("file={}, tstart={}, source_name={}, fch1={}, foff={}, nchans={}"
                            .format(os.path.basename(obj.path_dat), obj.tstart, obj.source_name,
                                    obj.fch1, obj.foff, obj.nchans))
            return None
        logger.info(f"Source '{on_source_complex_cadence}' is in this complex cadence.")
    else:
        matcher = path_record[0]

    # Display path_record rows.
    flag_terminate = False
    for obj in path_record:
        logger.info("file={}, tstart={}, source_name={}, fch1={}, foff={}, nchans={}"
                    .format(os.path.basename(obj.path_dat), obj.tstart, obj.source_name,
                            obj.fch1, obj.foff, obj.nchans))
        if on_source_complex_cadence: # Complex cadence?
            # If not a part of the complex cadence, then skip it.
            if on_source_complex_cadence != obj.source_name:
                continue
        # Part of the cadence, complex or not.
        # Make sure that the frequency range makes sense.
        ###print(f"DEBUG fch1 {obj.fch1}:{matcher.fch1}   foff {obj.foff}:{matcher.foff}   nchans {obj.nchans}:{matcher.nchans}")
        if not close_enough(obj.fch1, matcher.fch1) \
        or not close_enough(obj.foff, matcher.foff) \
        or obj.nchans != matcher.nchans:
            logger.error("Inconsistent frequency range!  This does not look like a cadence of related files.")
            flag_terminate = True
    if flag_terminate:
        return None


    # If this is a complex cadence,
    # * construct a complex_cadence list of 1s and 0s.
    # * compute count_cadence = number of matches on on_source_complex_cadence.
    if on_source_complex_cadence:
        complex_cadence = []
        count_cadence = 0
        for i in range(0, len(source_name_list)):
            source = source_name_list[i]
            if source == on_source_complex_cadence:
                complex_cadence.append(1)
                count_cadence += 1
            else:
                complex_cadence.append(0)
        if count_cadence > 0:
            print("The derived complex cadence is: " + str(complex_cadence))
        else:
            logger.error(f"Sorry, no potential candidates with your given on_source_complex_cadence={on_source_complex_cadence}  :(")
            return None

    num_of_sets = int(n_files / number_in_cadence)
    print("There are " + str(len(dat_file_list)) + " total files in the filelist "
          + dat_file_list_str)
    print("Therefore, looking for events in " + str(num_of_sets) + " on-off set(s)")

    if filter_threshold == 1:
        print("Present in an ON source only")
    if filter_threshold == 2:
        print("Present in at least one ON source with RFI rejection from the OFF sources")
    if filter_threshold == 3:
        print("Present in all ON sources with RFI rejection from the OFF sources")

    if not saving:
        print("A CSV file will not be saved")
    else:
        print("A CSV file will be saved")

    if user_validation:
        question = "Do you wish to proceed with these settings?"
        while "the answer is invalid":
            reply = str(input(question+' (y/n): ')).lower().strip()
            if reply == '':
                return None
            if reply[0] == 'y':
                break
            if reply[0] == 'n':
                return None

    #Looping over number_in_cadence chunks.
    candidate_list = []
    for ii in range(num_of_sets):
        sublist_low = number_in_cadence * ii
        sublist_high = sublist_low + number_in_cadence
        file_sublist = dat_file_list[sublist_low : sublist_high]
        if not complex_cadence:
            if on_off_first == 'ON':
                filename = os.path.basename(file_sublist[0])
            else: # on_off_first == 'OFF'
                filename = os.path.basename(file_sublist[1])
        else: # complex_cadence
            filename = os.path.basename(file_sublist[complex_cadence.index(1)])

        print("=== First DAT file in set:  " + filename + " ===")
        cand = find_events(file_sublist,
                           SNR_cut=SNR_cut,
                           min_drift_rate=min_drift_rate,
                           max_drift_rate=max_drift_rate,
                           check_zero_drift=check_zero_drift,
                           filter_threshold=filter_threshold,
                           on_off_first=on_off_first,
                           complex_cadence=complex_cadence)
        cand_len = 1
        if cand is None:
            cand_len = 0
        if cand_len != 0:
            candidate_list.append(cand)
    if len(candidate_list) > 0:
        find_event_output_dataframe = pd.concat(candidate_list)
    else:
        logger.error("Sorry, no potential candidates with your given parameters :(")
        return None

    print("===========   find_event_pipeline: output dataframe is complete   ===========")

    if saving:
        if csv_name is None:
            prefix = os.path.dirname(dat_file_list[0]) + '/' + source_name_list[0]
            if check_zero_drift:
                filestring = prefix + '_f' + str(filter_threshold) + '_snr' \
                    + str(SNR_cut) + '_zero' + '.csv'
            else:
                filestring = prefix + '_f' + str(filter_threshold) + '_snr' \
                    + str(SNR_cut) + '.csv'
        else:
            filestring = csv_name
        if not isinstance(find_event_output_dataframe, list):
            find_event_output_dataframe.to_csv(filestring)
            print("find_event_pipeline: Saved CSV file to {}".format(filestring))
        else:
            logger.error("Sorry, no events to save :(")
            return None

    return find_event_output_dataframe
