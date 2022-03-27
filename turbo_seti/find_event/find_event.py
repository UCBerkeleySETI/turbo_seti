#!/usr/bin/env python
r"""
Backend script to find drifting, narrowband events in a generalized cadence of
radio SETI observations (any number of ons, any number of offs, any pattern -
streamlined for alternating on-off sequences).

The main function contained in this file is :func:`find_events` uses the other
helper functions in this file (described below) to read a list of turboSETI .dat
files. It then finds events within this group of files.
"""

import time
import re
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
#^To remove pandas warnings: default='warn'


def end_search(t0):
    r"""
    Ends the search when there are no candidates left, or when the filter
    level matches the user-specified level.

    Parameters
    ----------
    t0 : time

    """
    #Report elapsed search time
    t1 = time.time()
    print('find_events: Elapsed time: %.2f sec' % ((t1-t0)))


def read_dat(filename):
    r"""
    Read a turboseti .dat file.

    Parameters
    ----------
    filename : str
        Name of .dat file to open.

    Returns
    -------
    df_data : dict
        Pandas dataframe of hits.

    """
    file_dat = open(filename.strip())
    hits = file_dat.readlines()

    # Get info from the .dat file header
    FileID = hits[1].strip().split(':')[-1].strip()
    Source = hits[3].strip().split(':')[-1].strip()

    MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
    RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
    DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

    DELTAT = hits[5].strip().split('\t')[0].split(':')[-1].strip()  # s
    DELTAF = hits[5].strip().split('\t')[1].split(':')[-1].strip()  # Hz

    # As of version 2.1.0, there are 2 additional parameters:
    try:
        max_drift_rate = hits[5].strip().split('\t')[2].split(':')[-1].strip() # Hz/s
        obs_length = hits[5].strip().split('\t')[3].split(':')[-1].strip() # s
    except Exception as exc:
        msg = '*** Out-of-date DAT file format detected in {}.  Rerun doppler search!' \
              .format(filename)
        raise RuntimeError(msg) from exc

    # Get info from individual hits (the body of the .dat file)
    all_hits = []
    for hit_line in hits[9:]:
        hit_fields = re.split(r'\s+', re.sub(r'[\t]', ' ', hit_line).strip())
        all_hits.append(hit_fields)

    # Now reorganize that info to be grouped by column (parameter)
    # not row (individual hit)
    if all_hits:
        TopHitNum = list(zip(*all_hits))[0]
        DriftRate = [float(df) for df in list(zip(*all_hits))[1]]
        SNR = [float(ss) for ss in list(zip(*all_hits))[2]]
        Freq = [float(ff) for ff in list(zip(*all_hits))[3]]
        ChanIndx = list(zip(*all_hits))[5]
        FreqStart = list(zip(*all_hits))[6]
        FreqEnd = list(zip(*all_hits))[7]
        CoarseChanNum = list(zip(*all_hits))[10]
        FullNumHitsInRange = list(zip(*all_hits))[11]

        data = {'TopHitNum': TopHitNum,
                'DriftRate': DriftRate,
                'SNR': SNR,
                'Freq': Freq,
                'ChanIndx': ChanIndx,
                'FreqStart': FreqStart,
                'FreqEnd': FreqEnd,
                'CoarseChanNum': CoarseChanNum,
                'FullNumHitsInRange': FullNumHitsInRange
                }

        # Creating pandas dataframe from data we just read in
        df_data = pd.DataFrame(data)
        df_data = df_data.apply(pd.to_numeric)

    else:
        df_data = pd.DataFrame()

    # Matching column information from before to the .dat data we read in
    df_data['FileID'] = FileID
    df_data['Source'] = Source.upper()
    df_data['MJD'] = MJD
    df_data['RA'] = RA
    df_data['DEC'] = DEC
    df_data['DELTAT'] = DELTAT
    df_data['DELTAF'] = DELTAF

    # As of 2.1.0, 2 new dataframe columns:
    df_data['max_drift_rate'] = max_drift_rate
    df_data['obs_length'] = obs_length

    # Adding extra columns that will be filled out by this program
    df_data['Hit_ID'] = ''
    df_data['status'] = ''
    df_data['in_n_ons'] = ''
    df_data['RFI_in_range'] = ''

    return df_data


def calc_freq_range(hit, delta_t=0.0, max_dr=True, follow=False):
    r"""
    Calculates a range of frequencies where RFI in an off-source could
    be related to a hit in an on-source, given a freq and drift_rate.

    Parameters
    ----------
    hit : dict
    delta_t : float, optional
    max_dr : bool, optional
    follow : bool, optional

    Returns
    -------
    [low_bound, high_bound] : list

    """
    if max_dr:
        drift_rate = float(hit['max_drift_rate'])
    else:
        drift_rate = float(hit['DriftRate'])

    if drift_rate == 0.0:
        drift_rate = float(hit['DELTAF'])/float(hit['DELTAT'])
    if follow:
        freq = float(hit['Freq']) + drift_rate*(delta_t)/1e6
        delta_t = 2.0 * float(hit['obs_length'])  # NOTE: I'm doing twice the amount in order to widen the range a bit.
    else:
        freq = float(hit['Freq'])
        delta_t = delta_t + float(hit['obs_length'])  #adding to the end of the obs.

    low_bound  = freq - abs(drift_rate)*delta_t/1e6
    high_bound = freq + abs(drift_rate)*delta_t/1e6

    return [low_bound,high_bound]


def follow_event(hit, on_table, get_count=True):
    r"""
    Follows a given hit to the next observation of the same target and
    looks for hits which could be part of the same event.

    Parameters
    ----------
    hit : dict
    on_table : dict
    get_count : bool

    Returns
    -------
    new_on_table or count : dict or int

    """

    #uses calc_freq_range to see how much the hit *should* have drifted by
    freq_range = calc_freq_range(hit, delta_t=on_table['delta_t'].values[0], max_dr=False, follow=True)

    #looks at the on (next given observation) to see if there are any
    #hits that could plausibly be related to the first one
    new_on_table = on_table[(on_table['Freq']>freq_range[0]) & (on_table['Freq']<freq_range[1])]

    #we only care if there were or were not plausible hits, so turn output into
    #a binary variable with get_count=True
    if get_count:
        n_hits_in_range = len(new_on_table)

        #Avoiding cases where multiple hits in one obs, and none in the other.
        if n_hits_in_range:
            return 1
        return 0

    return new_on_table


def not_yet_seen(mylist, argument):
    """
    Search a list to see if argument is already there.

    Parameters
    ----------
    mylist : list
        List of things that have been already seen.
    argument : int
        An integer to add to list if not alreay seen.

    Returns
    -------
    bool
       True :: Not yet seen so the argument was added.
       False :: Already seen.
    """
    for ii in mylist:
        if ii == argument:
            return False

    mylist.insert(0, argument)
    return True



def find_events(dat_file_list, check_zero_drift=False, filter_threshold=3,
                on_off_first='ON', complex_cadence=False,
                SNR_cut=None, min_drift_rate=None, max_drift_rate=None):
    r"""
    Reads a list of turboSETI .dat files.

    Parameters
    ----------
    dat_file_list : list
        A Python list of .dat files with ON observations of a
        single target alternating with OFF observations. This
        cadence can be of any length, given that the ON source
        is every other file. This includes Breakthrough Listen
        standard ABACAD as well as OFF first cadences like
        BACADA. Minimum cadence length is 2, maximum cadence
        length is unspecified (currently tested up to 6).
    check_zero_drift : bool, optional
        A True/False flag that tells the program whether to
        include hits that have a drift rate of 0 Hz/s. Earth-
        based RFI tends to have no drift rate, while signals
        from the sky are expected to have non-zero drift rates.
        Default is False.
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
    on_off_first : str {'ON', 'OFF}, optional
        Tells the code whether the .dat sequence starts with
        the ON or the OFF observation. Valid entries are 'ON'
        and 'OFF' only.
    complex_cadence : bool, optional
        A Python list of 1s and 0s corresponding to which
        files in the file_sublist are on-sources and which are
        off_sources for complex (i.e. non alternating) cadences.
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
 
    Examples
    --------
    It is highly recommended that users interact with this program via the
    front-facing find_event_pipeline.py script. See the usage of that file in
    its own documentation.

    If you would like to run find_events without calling
    find_event_pipeline.py, the usage is as follows:

    >>> find_event.find_events(file_sublist, SNR_cut=10, check_zero_drift=False,
    ...                        filter_threshold=3, on_off_first='ON', complex_cadence=False)

    Notes
    -----
    It calls other functions to find events within this group of files.
    Filter_threshold allows the return of a table of events with hits at
    different levels of filtering.
    Filter_threshold = [1,2,3] means:
        1) Hits above an SNR cut witout AB check
        2) Hits that are only in some As and no Bs
        3) Hits that are only in all As and no Bs

    """
    #Initializing timer
    t0 = time.time()

    print(f"find_events: SNR_cut = {SNR_cut}")
    print(f"find_events: min_drift_rate = {min_drift_rate}")
    print(f"find_events: max_drift_rate = {max_drift_rate}")
    print("find_events: Loading data...")

    #Preparing to read in the list of files
    on_table_list = []
    off_table_list = []

    off_count = 1
    on_count = 1

    odd_even_indicator = 0
    on_off_indicator = 0
    if complex_cadence: # One source name was provided.
        number_of_ons = 0
    else: # complex_cadence=False
        number_of_ons = int(np.floor(len(dat_file_list) / 2.0))
        if on_off_first == 'ON':
            odd_even_indicator = 1
            number_of_ons = int(np.ceil(len(dat_file_list) / 2.0))

    #reading in the list of files
    for ii, dat_file in enumerate(dat_file_list):
        if complex_cadence:
            on_off_indicator = int(complex_cadence[ii])
            number_of_ons += on_off_indicator
        #Checking if the file is an on or off observation
        #if off
        if (ii%2 == odd_even_indicator and complex_cadence == False) \
        or (on_off_indicator == 0 and complex_cadence != False):
            #Using read_dat function to read the .dat file
            #and create the pandas hit table for OFF sources
            off_table_ii = read_dat(dat_file)
            nhits = len(off_table_ii)
            print('find_events: Loaded %i hits from %s (OFF)'%(nhits, dat_file))
            if nhits > 0:
                off_table_ii['status'] = 'off_table_{}'.format(off_count)
                #Grouping all of the off hits into one table
                off_table_list.append(off_table_ii)
                off_count += 1

        else:
            #Using read_dat function to read the .dat file
            #and create the pandas hit table for ON sources
            on_table_ii = read_dat(dat_file)
            nhits = len(on_table_ii)
            print('find_events: Loaded %i hits from %s (ON)'%(nhits, dat_file))
            if nhits > 0:
                on_table_ii['status'] = 'on_table_{}'.format(on_count)
                #Grouping all of the on hits into one table
                on_table_list.append(on_table_ii)
                on_count += 1

    #If there are no hits on any on target, return to caller
    if len(on_table_list) < 1:
        print('*** find_events: There are no hits in this cadence (on_table_list) :(')
        end_search(t0)
        return None

    #Concatenating the on and off tables into a giant on table
    #and a giant off table
    on_table = pd.concat(on_table_list,ignore_index=True)
    if len(off_table_list) > 0:
        off_table = pd.concat(off_table_list,ignore_index=True)
    else: # nil list of OFF tables
        off_table = []

    #Check that all targets in the on_table come from the same source
    #Fix issue where some sources have B'' format
    uniqlist = list(on_table['Source'].unique())
    for i, ele in enumerate(uniqlist):
        try:
            uniqlist[i] = ele.strip("'B")
        except:
            continue
    if len(set(uniqlist)) > 1:
        raise ValueError('find_events: There are multiple sources in the ON table.'
                         'Please check your input files, '
                         'on_off_first parameter,'
                         'or complex_cadence.')

    #Obtain the start times for each hit in the first on table
    ref_time = float(on_table[on_table['status'] == 'on_table_1']['MJD'].unique()[0])

    #Calculating and saving delta_t, in seconds, to follow a given hit from
    #the first ON table to see if it appears in subsequent ON tables
    on_table['delta_t'] = on_table['MJD'].apply(lambda x:
        (float(x) - ref_time)*3600*24)

    #######################################################################
    print('find_events: All data loaded!')
    print()
    print('find_events: Finding events in this cadence...')
    #######################################################################
    # Using logic statements and pandas capabilities to find events that
    # meet these requirements:
    # * If zero drift rate checking is requested, then filter out hits with
    #      a drift rate exactly = 0.0.  Obsolete parameter?
    # * SNR value > SNR cut.
    # * Drift rate value > min_drift_rate.
    # * Drift rate value < max_drift_rate.
    # * Passed level 1 AND that are in at least one ON but no OFFs.
    # * Passed level 2 AND that are present in *ALL* ONs.

    #Optionally remove signals that don't have a drift rate
    if check_zero_drift:
        zero_adjusted_table = on_table
    else:
        zero_adjusted_table = on_table[on_table['DriftRate'] != 0.0]

    # If SNR_cut is None, include all signals from the zero adjusted table.
    # Else, include only signals with SNRs above the threshold value (SNR_cut).
    if SNR_cut is None:
        filter_adjusted_table = zero_adjusted_table
    else:
        filter_adjusted_table = zero_adjusted_table[ abs(zero_adjusted_table['SNR']) > abs(float(SNR_cut)) ]

    # If min_drift_rate is None, include all signals from the filter_adjusted_table.
    # Else, include only signals with a drift_rate above min_drift_rate.
    if min_drift_rate is not None:
        filter_adjusted_table = filter_adjusted_table[abs(filter_adjusted_table['DriftRate']) > abs(float(min_drift_rate))]

    # If max_drift_rate is None, include all signals from the filter_adjusted_table.
    # Else, include only signals with a drift_rate below max_drift_rate.
    if max_drift_rate is not None:
        filter_adjusted_table = filter_adjusted_table[abs(filter_adjusted_table['DriftRate']) < abs(float(max_drift_rate))]

    #If there are no hits left once SNR adjustments are made,
    #let the user know, return the zero-drift adjusted table, and end the
    #search
    if len(filter_adjusted_table) == 0:
        print('*** find_events: The filter_adjusted_table is empty.  No hits. :(')
        end_search(t0)
        return None
    if filter_threshold == 1:
        print('find_events (Filter threshold 1): Found a total of %i hits in this cadence!'%len(filter_adjusted_table))
        end_search(t0)
        return filter_adjusted_table
    print('find_events (Filter threshold 2/3): Found a total of %i hits in this cadence!'%len(filter_adjusted_table))

    #----------------------------------------------------------------------

    #Now find how much RFI is within a frequency range of the hit
    #by comparing the ON to the OFF observations. Update RFI_in_range
    if len(off_table) == 0:
        print('find_events: Length of off_table = 0')
        filter_adjusted_table['RFI_in_range'] = 0
    else:
        filter_adjusted_table['RFI_in_range'] = filter_adjusted_table.apply(
            lambda hit:
                len(off_table[((off_table['Freq'] > calc_freq_range(hit)[0])
                               & (off_table['Freq'] < calc_freq_range(hit)[1])
                               )]), axis=1)

    #If there is no RFI in range of the hit, it graduates to the
    #not_in_B_table
    not_in_off_table = filter_adjusted_table[filter_adjusted_table['RFI_in_range'] == 0]

    if len(not_in_off_table) == 0:
        print('find_events (Filter threshold 2/3): Found no hits present in only the on observations for this cadence :(')
        end_search(t0)
        return None
    if filter_threshold == 2:
        print('find_events (Filter threshold 2): Found a total of %i hits in only the on observations for this cadence!'%len(not_in_off_table))
        end_search(t0)
        return not_in_off_table

    print('find_events: Found a total of %i hits in only the on observations for this cadence!'%len(not_in_off_table))

    #----------------------------------------------------------------------

    on_but_not_off_table_list = []
    in_all_ons_table = []
    #Follow the drifting of a hit to find the events that
    #are present in ALL on observations

    for i in range(1, number_of_ons + 1):
        on_but_not_off_table_list.append(not_in_off_table[not_in_off_table['status'] == 'on_table_{}'.format(i)])
    empty_counter = 0
    for hit_list in on_but_not_off_table_list:
        if hit_list.empty is True:
            empty_counter += 1
    if empty_counter == 0:
        first_on = on_but_not_off_table_list[0]


        def hit_func(hit):
            """
            Embedded hit_func()
            Args:
              hit:

            Returns:
                counter, int
            """
            counter = 0
            for ii in range(1, len(on_but_not_off_table_list)):
                counter += follow_event(hit, on_but_not_off_table_list[ii])
            return counter


        first_on['in_n_ons'] = first_on.apply(hit_func, axis=1)
        in_all_ons_table = first_on[first_on['in_n_ons'] == number_of_ons - 1]

        #Create list of events.
        filter_3_event_list = []

        # Create an event table of unique entries.
        singularities = []
        for hit_index, hit in in_all_ons_table.iterrows():
            for table in on_but_not_off_table_list:
                temp_table = follow_event(hit, table, get_count=False)
                temp_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
                if not_yet_seen(singularities, temp_table.iloc[0]['TopHitNum']):
                    filter_3_event_list += [temp_table]

    else:
        print('*** find_events (Filter threshold 3): At least one of the ON tables is empty :(')
        end_search(t0)
        return None

    if len(in_all_ons_table) > 0:
        best_events_table = pd.concat(filter_3_event_list)
        print('find_events: Found a total of {} events across this cadence!'.format(len(best_events_table)))
        end_search(t0)
        return best_events_table

    print('*** find_events (Filter threshold 3): Found no events across this cadence (in_all_ons_table empty)) :(')
    end_search(t0)
    return None
