#!/usr/bin/env python
"""
Part of the Breakthrough Listen software package turboSETI

Backend script to find drifting, narrowband events in a generalized cadence of
radio SETI observations (any number of ons, any number of offs, any pattern -
streamlined for alternating on-off sequences).

In this code, the following terminology is used:
Hit = single strong narrowband signal in an observation
Event = a strong narrowband signal that is associated with multiple hits
        across ON observations
    
The main function contained in this file is *find_events*
    Find_events uses the other helper functions in this file (described below)
    to read a list of turboSETI .dat files. It then finds events within this 
    group of files. 
    
The following helper functions are contained in this file:
    end_search      - prints the runtime and ends the program
    make_table      - generates a pandas dataframe from a turboSETI
                      output .dat file created by seti_event.py
    calc_freq_range - Calculates a range of frequencies where RFI in an OFF 
                      could be related to a hit in an ON given a freq and 
                      drift_rate
    follow_event    - Follows a given hit to the next observation of the ON
                      target and looks for hits which could be part of the 
                      same event
                      
Usage (beta):
    It is highly recommended that users interact with this program via the
    front-facing find_event_pipeline.py script. See the usage of that file in
    its own documentation. 
    
    If you would like to run find_events without calling
    find_event_pipeline.py, the usage is as follows:
    
    find_event.find_events(file_sublist, 
                           SNR_cut=10, 
                           check_zero_drift=False, 
                           filter_threshold=3, 
                           on_off_first='ON',
                           complex_cadence=False)
    
    file_sublist        A Python list of .dat files with ON observations of a
                        single target alternating with OFF observations. This 
                        cadence can be of any length, given that the ON source 
                        is every other file. This includes Breakthrough Listen 
                        standard ABACAD as well as OFF first cadences like 
                        BACADA. Minimum cadence length is 2, maximum cadence 
                        length is unspecified (currently tested up to 6).
                   
    SNR_cut             The threshold SNR below which hits in the ON source 
                        will be disregarded. For the least strict thresholding, 
                        set this parameter equal to the minimum-searched SNR 
                        that you used to create the .dat files from 
                        seti_event.py. Recommendation (and default) is 10.
                   
    check_zero_drift    A True/False flag that tells the program whether to
                        include hits that have a drift rate of 0 Hz/s. Earth-
                        based RFI tends to have no drift rate, while signals
                        from the sky are expected to have non-zero drift rates.
                        Default is False.
                        
    filter_threshold    Specification for how strict the hit filtering will be.
                        There are 3 different levels of filtering, specified by
                        the integers 1, 2, and 3. Filter_threshold = 1 
                        returns hits above an SNR cut, taking into account the
                        check_zero_drift parameter, but without an ON-OFF check.
                        Filter_threshold = 2 returns hits that passed level 1
                        AND that are in at least one ON but no OFFs. 
                        Filter_threshold = 3 returns events that passed level 2
                        AND that are present in *ALL* ONs. Default is 3.
                        
    on_off_first        Tells the code whether the .dat sequence starts with
                        the ON or the OFF observation. Valid entries are 'ON'
                        and 'OFF' only. Default is 'ON'.
                        
    complex_cadence     A Python list of 1s and 0s corresponding to which
                        files in the file_sublist are on-sources and which are
                        off_sources for complex (i.e. non alternating) cadences.
                        Default is False.
                    
author: 
    Version 2.0 - Sofia Sheikh (ssheikhmsa@gmail.com)
    Version 1.0 - Emilio Enriquez (jeenriquez@gmail.com)
    
Last updated: 04/08/2020

***
NOTE: This code works for .dat files that were produced by seti_event.py
after turboSETI version 0.8.2, and blimpy version 1.1.7 (~mid 2019). The 
drift rates *before* that version were recorded with the incorrect sign
and thus the drift rate sign would need to be flipped in the make_table 
function.
***

"""

import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None  
#^To remove pandas warnings: default='warn'

#------
#Hardcoded values - SZS NOTE: Make these NOT Hardcoded
MAX_DRIFT_RATE = 2.0    # NOTE: these two values need to be updated.
OBS_LENGTH = 300.
#------
def end_search(t0):
    """ends the search when there are no candidates left, or when the filter
    level matches the user-specified level

    Args:
      t0: 

    Returns:

    """
    #Report elapsed search time
    t1 = time.time()
    print('Search time: %.2f sec' % ((t1-t0)))
    print('------   o   -------')
    return

def read_dat(filename):
    """Read a turboseti .dat file

    Args:
      filename(str): Name of .dat file to open

    Returns:
        pandas dataframe of hits
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

    # Get info from individual hits (the body of the .dat file)
    all_hits = [hit.strip().split('\t') for hit in hits[9:]]

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
        Fscrunch = list(zip(*all_hits))[12]

        data = {'TopHitNum': TopHitNum,
                'DriftRate': DriftRate,
                'SNR': SNR,
                'Freq': Freq,
                'ChanIndx': ChanIndx,
                'FreqStart': FreqStart,
                'FreqEnd': FreqEnd,
                'CoarseChanNum': CoarseChanNum,
                'FullNumHitsInRange': FullNumHitsInRange,
                'Fscrunch': Fscrunch
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

    # Adding extra columns that will be filled out by this program
    df_data['Hit_ID'] = ''
    df_data['status'] = ''
    df_data['in_n_ons'] = ''
    df_data['RFI_in_range'] = ''

    return df_data

def make_table(filename, init=False):
    """Creates a pandas dataframe with column names standard for turboSETI .dat
    output files, either directly (if) or by reading the file line-by line and
    then reorganizing the output (else)

    Args:
      filename: 
      init:  (Default value = False)

    Returns:
        df_data: pandas dataframe

    """
    
    if init:
        columns = ['FileID','Source','MJD','RA','DEC', 'DELTAT','DELTAF',
                   'TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 
                   'FreqStart', 'FreqEnd', 'CoarseChanNum', 
                   'FullNumHitsInRange','status','Hit_ID','in_n_ons',
                   'RFI_in_range']
        df_data = pd.DataFrame(columns=columns)

    else:
        df_data = read_dat(filename)
    return df_data

def calc_freq_range(hit,delta_t=0,max_dr=True,follow=False):
    """Calculates a range of frequencies where RFI in an off-source could
        be related to a hit in an on-source given a freq and drift_rate.

    Args:
      hit: 
      delta_t:  (Default value = 0)
      max_dr:  (Default value = True)
      follow:  (Default value = False)

    Returns:
        list [low_bound, high_bound]
    """
    if max_dr:
        drift_rate = MAX_DRIFT_RATE
    else:
        drift_rate = hit['DriftRate']
        
    if drift_rate == 0.0:
        drift_rate = float(hit['DELTAF'])/float(hit['DELTAT'])
    if follow:
        freq = hit['Freq'] + drift_rate*(delta_t)/1e6
        delta_t = 2*OBS_LENGTH  # NOTE: I'm doing twice the amount. To widen the range a bit. 
                                # Still it is dependent on hit['DriftRate']
    else:
        freq = hit['Freq']
        delta_t = delta_t+OBS_LENGTH  #adding to the end of the obs.
    
    low_bound  = freq - abs(drift_rate)*delta_t/1e6
    high_bound = freq + abs(drift_rate)*delta_t/1e6

    return [low_bound,high_bound]

def follow_event(hit,on_table,get_count=True):
    """Follows a given hit to the next observation of the same target and
    looks for hits which could be part of the same event.

    Args:
      hit: 
      on_table: 
      get_count:  (Default value = True)

    Returns:
        new_on_table or count

    """

    #uses calc_freq_range to see how much the hit *should* have drifted by
    freq_range = calc_freq_range(hit,delta_t=on_table['delta_t'].values[0],max_dr=False,follow=True)
    
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
        else:
            return 0
    else:
        return new_on_table

def find_events(dat_file_list,  SNR_cut=10, check_zero_drift=False, filter_threshold=3, on_off_first='ON', complex_cadence=False):
    """Reads a list of turboSETI .dat files.
        It calls other functions to find events within this group of files.
        Filter_threshold allows the return of a table of events with hits at
        different levels of filtering.
        Filter_threshold = [1,2,3] means:
            1) Hits above an SNR cut witout AB check
            2) Hits that are only in some As and no Bs
            3) Hits that are only in all As and no Bs

    Args:
      dat_file_list: 
      SNR_cut:  (Default value = 10)
      check_zero_drift:  (Default value = False)
      filter_threshold:  (Default value = 3)
      on_off_first:  (Default value = 'ON')

    Returns:

    """
    #Initializing timer
    t0 = time.time()
    
    print('------   o   -------')
    print("Loading data...")
    
    #Preparing to read in the list of files
    on_table_list = []
    off_table_list = []
    
    off_count = 1 
    on_count = 1  
        
    odd_even_indicator = 0
    on_off_indicator = 0
    if complex_cadence == False:
        number_of_ons = int(np.floor(len(dat_file_list) / 2.0))
        if on_off_first == 'ON':
            odd_even_indicator = 1
            number_of_ons = int(np.ceil(len(dat_file_list) / 2.0))  
    
    #reading in the list of files 
    for i,dat_file in enumerate(dat_file_list):
        if complex_cadence != False:
            on_off_indicator = int(complex_cadence[i])
        #Checking if the file is an on or off observation
        #if off
        if (i%2 == odd_even_indicator and complex_cadence == False) or (on_off_indicator == 0 and complex_cadence != False):
            #Using make_table function to read the .dat file 
            #and create the pandas hit table for off sources
            off_table_i = make_table(dat_file)
            nhits = len(off_table_i)
            print('Loaded %i hits from %s (OFF)'%(nhits, dat_file))
            if nhits > 0:
                off_table_i['status'] = 'off_table_%i'%off_count
                #Grouping all of the off hits into one table
                off_table_list.append(off_table_i)
                off_count+=1
            
        #if on
        else: 
            #Using make_table function to read the .dat file 
            #and create the pandas hit table for on sources
            on_table_i = make_table(dat_file)
            nhits = len(on_table_i)
            print('Loaded %i hits from %s (ON)'%(nhits, dat_file))
            if nhits > 0:
                on_table_i['status'] = 'on_table_%i'%on_count
                #Grouping all of the on hits into one table
                on_table_list.append(on_table_i)
                on_count+=1
                    
    #If there are no hits on any on target, end the program
    if not len(on_table_list):
        print('There are no hits in this cadence :(')
        end_search(t0)
        return
    
    #Concatenating the on and off tables into a giant on table 
    #and a giant off table
    on_table = pd.concat(on_table_list,ignore_index=True)
    off_table = pd.concat(off_table_list,ignore_index=True)        

    #Check that all targets in the on_table come from the same source
    #Fix issue where some sources have B'' format
    uniqlist = list(on_table['Source'].unique())
    for i, ele in enumerate(uniqlist):
        try:
            uniqlist[i] = ele.strip("'B")
        except:
            continue
    if len(set(uniqlist)) > 1:
        raise ValueError('There are multiple sources in the on table.' 
                         'Please check your input files, ' 
                         'on_off_first parameter,'
                         'or complex_cadence.')
    
    #Obtain the start times for each hit in the first on table
    ref_time = float(on_table[on_table['status'] == 'on_table_1']['MJD'].unique()[0])
    #Calculating and saving delta_t, in seconds, to follow a given hit from 
    #the first on table to see if it appears in the following on tables
    on_table['delta_t'] = on_table['MJD'].apply(lambda x: 
        (float(x) - ref_time)*3600*24)

    #######################################################################
    print('All data loaded!')
    print()
    print('Finding events in this cadence...')
    #######################################################################
    #Using logic statements and pandas capabilities to find events that:
    # 1) Are above an SNR cut, taking into account the check_zero_drift parameter, 
    #    but without an ON-OFF check.
    # 2) Passed level 1 AND that are in at least one ON but no OFFs 
    # 3) Passed level 2 AND that are present in *ALL* ONs.

    #Optionally remove signals that don't have a drift rate
    if check_zero_drift:
        zero_adjusted_table = on_table
    else:
        zero_adjusted_table = on_table[on_table['DriftRate'] != 0.0]

    #Remove signals below a certain signal-to-noise ratio (SNR_cut)
    snr_adjusted_table = zero_adjusted_table[zero_adjusted_table['SNR'] > SNR_cut]
    
    #If there are no hits left once SNR adjustments are made, 
    #let the user know, return the zero-drift adjusted table, and end the
    #search
    if (len(snr_adjusted_table) == 0):
        print('Found no hits above the SNR cut :(')
        end_search(t0)
        return
    if filter_threshold == 1:
        print('Found a total of %i hits above the SNR cut in this cadence!'%len(snr_adjusted_table))
        print('Filter level is 1 - returning this table...')
        end_search(t0)
        return snr_adjusted_table
    else:
        print('Found a total of %i hits above the SNR cut in this cadence!'%len(snr_adjusted_table))
    
    #----------------------------------------------------------------------

    #Now find how much RFI is within a frequency range of the hit 
    #by comparing the ON to the OFF observations. Update RFI_in_range
    snr_adjusted_table['RFI_in_range'] = snr_adjusted_table.apply(lambda hit: len(off_table[((off_table['Freq'] > calc_freq_range(hit)[0]) & (off_table['Freq'] < calc_freq_range(hit)[1]))]),axis=1)
        
    #If there is no RFI in range of the hit, it graduates to the 
    #not_in_B_table
    not_in_off_table = snr_adjusted_table[snr_adjusted_table['RFI_in_range'] == 0]

    if (len(not_in_off_table) == 0):
        print('Found no hits present in only the on observations in this cadence :(')
        end_search(t0)
        return
    if filter_threshold == 2:    
        print('Found a total of %i hits in only the on observations in this cadence!'%len(not_in_off_table))
        print('Filter level is 2 - returning this table...')
        end_search(t0)
        return not_in_off_table
    else:
        print('Found a total of %i hits in only the on observations in this cadence!'%len(not_in_off_table))
        
    #----------------------------------------------------------------------
    
    on_but_not_off_table_list = []
    in_all_ons_table = []
    #Follow the drifting of a hit to find the events that 
    #are present in ALL on observations
    
    for i in range(1, number_of_ons + 1):
        on_but_not_off_table_list.append(not_in_off_table[not_in_off_table['status'] == 'on_table_%i'%i])
    empty_counter = 0
    for hit_list in on_but_not_off_table_list:
        if hit_list.empty == True:
            empty_counter += 1
    if empty_counter == 0:
        first_on = on_but_not_off_table_list[0]#
        def hit_func(hit):
            """

            Args:
              hit: 

            Returns:

            """
            val = 0
            for i in range(1, len(on_but_not_off_table_list)):
                val += follow_event(hit, on_but_not_off_table_list[i])
            return val
        
        first_on['in_n_ons'] = first_on.apply(hit_func, axis=1)
        in_all_ons_table = first_on[first_on['in_n_ons'] == number_of_ons - 1]
        
        #Create list of events.
        filter_3_event_list = []

        for hit_index, hit in in_all_ons_table.iterrows():
            for table in on_but_not_off_table_list:
                temp_table = follow_event(hit,table,get_count=False)
                temp_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
                filter_3_event_list += [temp_table]

    else:
        print('NOTE: At least one of the on tables is empty - no events across this cadence :(')
        end_search(t0)
        return

    if len(in_all_ons_table) > 0:
        best_events_table = pd.concat(filter_3_event_list)
        print('Found a total of %i events across this cadence!'%(int(len(best_events_table)/3)))
        end_search(t0)
        return best_events_table
    
    else:
        print('NOTE: Found no events across this cadence :(')
        end_search(t0)
        return
    
    #----------------------------------------------------------------------
