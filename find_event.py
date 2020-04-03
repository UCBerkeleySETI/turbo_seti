#!/usr/bin/env python
'''
Script to find events in a group of ON-OFF observations.
Any signal found only in the ON is checked to be found in all the 3 ONs 
following the frequency drift of the signal.

Hit = single strong narrowband signal in an observation
Event = a strong narrowband signal that is associated with multiple hits
        across ON observations

Usage (beta):
    SZS: Working on this
author: 
    Version 2.0 - Sofia Sheikh (szs714@psu.edu), 
    Version 1.0 - Emilio Enriquez (jeenriquez@gmail.com)

***
NOTE: This code only works for .dat files that were produced by turboSETI
after turboSETI version 0.8.2, and blimpy version 1.1.7 (~mid 2019). The 
drift rates before that version were recorded with the incorrect sign
and thus the drift rate sign would need to be flipped in the make_table 
function.
***

'''

import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None  
#^To remove pandas warnings: default='warn'

#------
#Hardcoded values - SZS NOTE: Make these NOT Hardcoded
MAX_DRIFT_RATE = 2.0    # NOTE: these two values needs to be updated.
OBS_LENGTH = 300.
#------


def make_table(filename,init=False):
    ''' Creates a pandas dataframe from a turboSETI .dat output file.
    '''
    
    #creates a pandas dataframe with column names standard for turboSETI .dat
    #output files, either directly (if) or by reading the file line by line and
    #then reorganizing the output (else)
    
    if init:
        columns = ['FileID','Source','MJD','RA','DEC', 'DELTAT','DELTAF',
                   'TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 
                   'FreqStart', 'FreqEnd', 'CoarseChanNum', 
                   'FullNumHitsInRange','status','Hit_ID','in_n_ons',
                   'RFI_in_range']
        df_data = pd.DataFrame(columns=columns)

    else:
        file_dat = open(filename.strip())
        hits = file_dat.readlines()

        #Get info from the .dat file header
        FileID = hits[1].strip().split(':')[-1].strip()
        Source = hits[3].strip().split(':')[-1].strip()

        MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
        RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
        DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

        DELTAT = hits[5].strip().split('\t')[0].split(':')[-1].strip()   # s
        DELTAF = hits[5].strip().split('\t')[1].split(':')[-1].strip()   # Hz

        #Get info from individual hits (the body of the .dat file)
        all_hits = [hit.strip().split('\t') for hit in hits[9:]]
        
        #Now reorganize that info to be grouped by column (parameter) 
        #not row (individual hit)
        if all_hits:
            import numpy
            print(numpy.shape(list(zip(*all_hits))))
            TopHitNum = list(zip(*all_hits))[0]
            DriftRate = [float(df) for df in list(zip(*all_hits))[1]]
            SNR = [float(ss) for ss in list(zip(*all_hits))[2]]
            Freq = [float(ff) for ff in list(zip(*all_hits))[3]]
            ChanIndx = list(zip(*all_hits))[5]
            FreqStart = list(zip(*all_hits))[6]
            FreqEnd = list(zip(*all_hits))[7]
            CoarseChanNum = list(zip(*all_hits))[10]
            FullNumHitsInRange = list(zip(*all_hits))[11]

            data = {'TopHitNum':TopHitNum,
                    'DriftRate':DriftRate,
                    'SNR':SNR,
                    'Freq':Freq,
                    'ChanIndx':ChanIndx,
                    'FreqStart':FreqStart,
                    'FreqEnd':FreqEnd,
                    'CoarseChanNum':CoarseChanNum,
                    'FullNumHitsInRange':FullNumHitsInRange
                    }

            #Creating pandas dataframe from data we just read in
            df_data = pd.DataFrame(data)
            df_data = df_data.apply(pd.to_numeric)

        else:
            df_data = pd.DataFrame()

        #Matching column information from before to the .dat data we read in
        df_data['FileID'] = FileID
        df_data['Source'] = Source.upper()
        df_data['MJD'] = MJD
        df_data['RA'] = RA
        df_data['DEC'] = DEC
        df_data['DELTAT'] = DELTAT
        df_data['DELTAF'] = DELTAF

        #Adding extra columns that will be filled out by this program
        df_data['Hit_ID'] = ''
        df_data['status'] = ''
        df_data['in_n_ons'] = ''
        df_data['RFI_in_range'] = ''

    return df_data

def calc_freq_range(hit,delta_t=0,max_dr=True,follow=False):
    '''Calculates a range of frequencies where RFI in an off-source could
        be related to a hit in an on-source given a freq and drift_rate.
    '''
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

def follow_event(hit,A_table,get_count=True):
    ''' Follows a given hit to the next observation of the same target and 
    looks for hits which could be part of the same event.
    '''
    
    #uses calc_freq_range to see how much the hit *should* have drifted by
    freq_range = calc_freq_range(hit,delta_t=A_table['delta_t'].values[0],max_dr=False,follow=True)
    #looks at the A_table (next given observation) to see if there are any
    #hits that could plausibly be related to the first one
    new_A_table = A_table[(A_table['Freq']>freq_range[0]) & (A_table['Freq']<freq_range[1])]

    #we only care if there were or were not plausible hits, so turn output into
    #a binary variable with get_count=True
    if get_count:
        n_hits_in_range = len(new_A_table)

        #Avoiding cases where multiple hits in one obs, and none in the other.
        if n_hits_in_range:
            return 1
        else:
            return 0
    else:
        return new_A_table

def find_events(dat_file_list,
                SNR_cut=10,
                check_zero_drift=False,
                filter_threshold=3,
                on_off_first='ON',
                number_in_sequence=6):
    ''' Reads a list of turboSETI .dat files stored in a .lst file.
        It calls other functions to find events within this group of files.
        Filter_threshold allows the return of a table of events with hits at 
        different levels of filtering.
        Filter_threshold = [1,2,3] means:
            1) Hits above an SNR cut witout AB check
            2) Hits that are only in some As and no Bs
            3) Hits that are only in all As and no Bs
        Available on-off sequences:
            3OFF (BAC)
            3ON (ABA)
            4OFF (BACA)
            4ON (ABAC)
            5OFF (BACAD)
            5ON (ABACA)
            6OFF (BACADA)
            6ON (ABACAD)
    '''
    #Initializing timer
    t0 = time.time()
    
#---------------------------------------------#
#            A.....A                          #
#           B......B                          #
#           odd number, starts with ON or OFF #
#---------------------------------------------#   
    #if number_in_sequence%2 == 1:
    #Preparing to read in the list of files
    A_table_list = []
    off_table_list = []
    kk = 1   # counter for table name.
    ll = 1   # counter for table name.
    
    on_off_indicator = 0
    number_of_As = int(np.floor(number_in_sequence / 2.0))
    if on_off_first == 'ON':
        on_off_indicator = 1
        number_of_As = int(np.ceil(number_in_sequence / 2.0))
    
    for i,dat_file in enumerate(dat_file_list):
        #Preparing to read hit data for each file
        print('Reading hits data for %s'%dat_file)
    
        #Checking if the file is an A or off observation, based on index
        if i%2 == on_off_indicator:
            #Using make_table function to read the .dat file 
            #and create the pandas hit table
            off_table_i=make_table(dat_file)
            off_table_i['status'] = 'off_table_%i'%ll
            print('There are %i hits on this file.'%len(off_table_i))

            #Grouping all of the B hits into one table
            off_table_list.append(off_table_i)
            ll+=1
            
        else: 
            #Using make_table function to read the .dat file 
            #and create the pandas hit table
            Ai_table=make_table(dat_file)
            Ai_table['status'] = 'A%i_table'%kk
            print('There are %i hits on this file.'%len(Ai_table))
    
            #Grouping all of the A hits into one table
            A_table_list.append(Ai_table)
            kk+=1
    
    #If there are no hits on the A target, end the program
    if not len(A_table_list) and kk == 1:
        print('Exiting, since A_table_list is empty.')
        t1 = time.time()
        print('Search time: %.2f sec' % ((t1-t0)))
        print('------   o   -------')
        return pd.DataFrame({'This is an empty Data Frame' : []})
    
    #Concatenating the A and B tables into a giant A table 
    #and a giant B table
    A_table = pd.concat(A_table_list,ignore_index=True)
    B_table = pd.concat(off_table_list,ignore_index=True)        
    
    #Check that all targets in the A_table come from the same source
    if A_table['Source'].unique().shape[0] > 1:
        raise ValueError('There are multiple sources in the A table.' 
                         'Please check your input files, ' 
                         'on_off_sequence parameter, ' 
                         'and number_in_sequence parameter.')
    
    #Obtain the start times for each hit in the first A table
    ref_time = float(A_table[A_table['status'] == 'A1_table']['MJD'].unique()[0])
    #Calculating and saving delta_t, in seconds, to follow a given hit from 
    #the first A table to see if it appears in the second and third
    A_table['delta_t'] = A_table['MJD'].apply(lambda x: 
        (float(x) - ref_time)*3600*24)

    #######################################################################
    print('FINDING ALL EVENTS')
    #######################################################################
    #Using logic statements and pandas capabilities to find events that are:
    # 1) Present in an A source (just the A-table) [Filter level 1] 
    # 2) Present in at least one A source but no B sources [Filter level 2]
    # 3) Present in all A sources but no B sources [Filter level 3]

    #Optionally remove signals that don't have a drift rate
    if check_zero_drift:
        zero_adjusted_table = A_table
    else:
        zero_adjusted_table = A_table[A_table['DriftRate'] != 0.0]

    #Remove signals below a certain signal-to-noise ratio (SNR_cut)
    snr_adjusted_table = zero_adjusted_table[zero_adjusted_table['SNR'] > SNR_cut]
    
    #If there are no hits left once SNR adjustments are made, 
    #let the user know, return the zero-drift adjusted table, and end the
    #search
    if (len(snr_adjusted_table) == 0):
        print('Found no hits above the SNR cut.')
        return zero_adjusted_table
    else:
        print('Found hits above the SNR cut!')
    
    #----------------------------------------------------------------------

    #Now find how much RFI is within a frequency range of the hit 
    #by comparing the ON to the OFF observations. Update RFI_in_range
    snr_adjusted_table['RFI_in_range'] = snr_adjusted_table.apply(lambda hit: len(B_table[((B_table['Freq'] > calc_freq_range(hit)[0]) & (B_table['Freq'] < calc_freq_range(hit)[1]))]),axis=1)
        
    #If there is no RFI in range of the hit, it graduates to the 
    #not_in_B_table
    not_in_off_table = snr_adjusted_table[snr_adjusted_table['RFI_in_range'] == 0]

    if (len(not_in_off_table) == 0):
        print('Found no hits present in only the A observations.')
        return snr_adjusted_table
    else:
        print('Found hits present in only the A observations!')
        
    #----------------------------------------------------------------------
    
    A_but_not_off_table_list = []
    in_all_As_table = []
    #Follow the drifting of a hit to find the events that 
    #are present in ALL A observations
    
    for i in range(1, number_of_As + 1):
        A_but_not_off_table_list.append(not_in_off_table[not_in_off_table['status'] == 'A%i_table'%i])
    empty_counter = 0
    for hit_list in A_but_not_off_table_list:
        if hit_list.empty == True:
            empty_counter += 1
    if empty_counter == 0:
        first_A = A_but_not_off_table_list[0]#
        def hit_func(hit):
            val = 0
            for i in range(1, len(A_but_not_off_table_list)):
                val += follow_event(hit, A_but_not_off_table_list[i])
            return val
        
        first_A['in_n_ons'] = first_A.apply(hit_func, axis=1)
        in_all_As_table = first_A[first_A['in_n_ons'] == number_of_As - 1]

    else:
        print('NOTE: Found no hits present in all A observations.')

    if len(in_all_As_table) > 0:
        print('NOTE: Found some events! :)')
    else:
        print('NOTE: Found no events. :(')
        return pd.concat(A_but_not_off_table_list)
    
    #----------------------------------------------------------------------


    #Create list of events.
    filter_3_event_list = []

    for hit_index, hit in in_all_As_table.iterrows():
        for table in A_but_not_off_table_list:
            temp_table = follow_event(hit,table,get_count=False)
            temp_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
            filter_3_event_list += [temp_table]

    AAA_table = pd.concat(filter_3_event_list)
    
    #######################################################################
    #######################################################################
    
    #If there are any Filter Level 3 candidates, report them!
    if len(AAA_table) > 0:
        print('Found: %2.2f events at Filter Level 3!'%(int(len(AAA_table)/3)))
    
    #Report elapsed search time
    t1 = time.time()
    print('Search time: %.2f sec' % ((t1-t0)))
    print('------   o   -------')
    
    #SZS: This is where returning happens, 
    #figure out a better way to do this
    return(AAA_table)
