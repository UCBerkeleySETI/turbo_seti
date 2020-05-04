#!/usr/bin/env python

"""
Front-facing script to find drifting, narrowband events in a set of generalized 
cadences of ON-OFF radio SETI observations.

In this code, the following terminology is used:
Hit = single strong narrowband signal in an observation
Event = a strong narrowband signal that is associated with multiple hits
        across ON observations
    
The main function contained in this file is *find_event_pipeline*
    Find_event_pipeline calls find_events from find_events.py to read a list 
    of turboSETI .dat files. It then finds events within this group of files. 
                      
Usage (beta):
    import find_event_pipeline;
    find_event_pipeline.find_event_pipeline(dat_file_list_str, 
                                            SNR_cut=10,
                                            check_zero_drift=False,
                                            filter_level=3, 
                                            on_off_first='ON', 
                                            number_in_cadence=6, 
                                            saving=True,  
                                            user_validation=False)
    
    dat_file_list_str   The string name of a plaintext file ending in .lst 
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
    
    number_in_cadence   The number of files in a single ON-OFF cadence.
                        Default is 6 for ABACAD.
                        
    saving              A True/False flag that tells the program whether to 
                        save the output array as a .csv
                        
    user_validation     A True/False flag that, when set to True, asks if the
                        user wishes to continue with their input parameters
                        (and requires a 'y' or 'n' typed as confirmation)
                        before beginning to run the program. Recommended when
                        first learning the program, not recommended for 
                        automated scripts.
                    
author: 
    Version 2.0 - Sofia Sheikh (ssheikhmsa@gmail.com), 
    Version 1.0 - Emilio Enriquez (jeenriquez@gmail.com)

***
NOTE: This code works for .dat files that were produced by seti_event.py
after turboSETI version 0.8.2, and blimpy version 1.1.7 (~mid 2019). The 
drift rates *before* that version were recorded with the incorrect sign
and thus the drift rate sign would need to be flipped in the make_table 
function.
***

"""

#required packages and programs
import find_event
import pandas as pd

#required for updated_find_event
import time
import numpy as np

def find_event_pipeline(dat_file_list_str,
                        SNR_cut=10, 
                        check_zero_drift=False, 
                        filter_threshold=3, 
                        on_off_first='ON', 
                        number_in_cadence=6, 
                        saving=True, 
                        user_validation=False): 
    """

    Args:
      dat_file_list_str: 
      SNR_cut:  (Default value = 10)
      check_zero_drift:  (Default value = False)
      filter_threshold:  (Default value = 3)
      on_off_first:  (Default value = 'ON')
      number_in_cadence:  (Default value = 6)
      saving:  (Default value = True)
      user_validation:  (Default value = False)

    Returns:

    """
    print()
    print("************   BEGINNING FIND_EVENT PIPELINE   **************")
    print()
    print("Assuming the first observation is an " + on_off_first)
    
    #Opening list of files
    dat_file_list = open(dat_file_list_str).readlines()
    dat_file_list = [files.replace('\n','') for files in dat_file_list]
    dat_file_list = [files.replace(',','') for files in dat_file_list]
    n_files = len(dat_file_list)
    
    print("There are " + str(len(dat_file_list)) + " total files in the filelist " + dat_file_list_str)
    print("therefore, looking for events in " + str(int(n_files/number_in_cadence)) + " on-off set(s)")
    print("with a minimum SNR of " + str(SNR_cut))
    
    if filter_threshold == 1:
        print("Present in an A source only, above SNR_cut")
    if filter_threshold == 2:
        print("Present in at least one A source with RFI rejection from the off-sources")
    if filter_threshold == 3:
        print("Present in all A sources with RFI rejection from the off-sources")
    
    if check_zero_drift == False:
        print("not including signals with zero drift")
    if check_zero_drift == True:
        print("including signals with zero drift")
    if saving == False:
        print("not saving the output files")
    if saving == True:
        print("saving the output files")    
    
    if user_validation == True:
        question = "Do you wish to proceed with these settings?"
        while "the answer is invalid":
            reply = str(input(question+' (y/n): ')).lower().strip()
            if reply == '':
                return
            if reply[0] == 'y':
                break
            if reply[0] == 'n':
                return
    
    #Looping over number_in_cadence chunks.
    candidate_list = []
    for i in range((int(n_files/number_in_cadence))):
        file_sublist = dat_file_list[number_in_cadence*i:((i*number_in_cadence)+(number_in_cadence))]
        if on_off_first == 'ON':
            name=file_sublist[0].split('_')[5]  
        if on_off_first == 'OFF':
            name=file_sublist[1].split('_')[5] 
        print()
        print("***       " + name + "       ***")
        print()
        cand = find_event.find_events(file_sublist, 
                                      SNR_cut=SNR_cut, 
                                      check_zero_drift=check_zero_drift, 
                                      filter_threshold=filter_threshold, 
                                      on_off_first=on_off_first)
        cand_len = 1
        if cand is None:
            cand_len = 0
        if cand_len != 0:
            candidate_list.append(cand)
    if len(candidate_list) > 0:
        find_event_output_dataframe = pd.concat(candidate_list)
    else:
        "Sorry, no potential candidates with your given parameters :("
        find_event_output_dataframe = []

    print("************  ENDING FIND_EVENT PIPELINE   **************")
    
    if saving == True:
        if check_zero_drift == True:
            filestring = name + '_f' + str(filter_threshold) + '_snr' + str(SNR_cut) + '_zero' + '.csv'
        else:
            filestring = name + '_f' + str(filter_threshold) + '_snr' + str(SNR_cut) + '.csv'
        
        find_event_output_dataframe.to_csv(filestring)


    return(find_event_output_dataframe)

#***********plotting code*********
#name_array = np.array(['HD_109376', 'HD_115153', 'HD_152956','LTT_7183', 'LTT_7494', 'GJ_757', 'BHIP_61711', 'HD_117004', 'LTT_6827','BHIP_88982', 'BHIP_93072', 'BHIP_96440'])
#    find_event_output_dataframe.to_csv(r'/home/ssheikh/ETZ_analysis/find_event_pipes/' + node_string + '_f' + str(filter_level) + '.csv')
#    print(".csv successfully saved")
#    trimmed_event_dataframe = find_event_output_dataframe.loc[:, ['Source', 'Freq', 'DriftRate','FreqStart']]
#    if len(find_event_output_dataframe) > 100:
#        print("too many events to plot, setting a stricter SNR threshold")
#        if node_string == 'blc24':
#            trimmed_event_dataframe = find_event_output_dataframe.loc[find_event_output_dataframe['SNR'] > 50, ['Source', 'Freq', 'DriftRate','FreqStart']]
#         
#        if node_string == 'blc25':
#            trimmed_event_dataframe = find_event_output_dataframe.loc[find_event_output_dataframe['SNR'] > 100, ['Source', 'Freq', 'DriftRate','FreqStart']]
#           
#        if node_string == 'blc34':
#                        trimmed_event_dataframe = find_event_output_dataframe.loc[find_event_output_dataframe['SNR'] > 20, ['Source', 'Freq', 'DriftRate','FreqStart']]
#           
#        print("now less than 100 events - plotting...")
#
#    #knowing whether a star is A or B, and what its indexes are
#    ab_array = np.zeros(len(name_array))
#    ab_array[:(len(name_array)/2)] = 1
#    index_array = range(0, len(dat_file_list), 6)
#
#    #Opening list of files, make them A and B lists
#    fil_file_list = open(fil_file_list_string).readlines()
#    fil_file_list = [files.replace('\n','') for files in fil_file_list]
#    fil_file_list = [files.replace(',','') for files in fil_file_list]
#    fil_file_list_b = fil_file_list[1:]
#    fil_file_list_b.append(fil_file_list[0])
#    fil_file_list = fil_file_list + fil_file_list_b

#    #loop through all target stars with promising candidates and plot them
#    for source in trimmed_event_dataframe.Source.unique():
#        print(source)
#        star_id_index = np.where(name_array == source)[0][0]
#        start_index = index_array[star_id_index]
        #plotting
#        plot_candidate_events_individually(trimmed_event_dataframe, correct_fils, source, node_string, filter_level, show=False, overwrite=False, offset=0)


#event_finder_and_plotter('blc00', 3, 10, #'/home/ssheikh/ETZ_analysis/sixes/A_lists/dats/blc/blc00_dats.lst', #'/home/ssheikh/ETZ_analysis/sixes/A_lists/fils/blc/blc00_fils.lst')


