#!/usr/bin/env python

#All star names must be entered in order, all A stars
#then all B-stars. In the future, I may automate this.
#node string = ex. 'blc00'
#filter_level = 1, 2, or 3
#SNR = the SNR cut for the analysis ex. 10
#dat_file_list_string = path to list of .dat files for
#    all of the candidates, with filename
#fil_file_list_string = path to list of .fil files for
#    all of the candidates, with filename

#required packages and programs
import find_event
import pandas as pd

#required for updated_find_event
import time
import numpy as np

def find_event_pipeline(filter_level, 
                        SNR, 
                        dat_file_list_string, 
                        on_off_first='ON', 
                        number_in_sequence=6, 
                        saving=False, 
                        zero_drift_parameter=True, 
                        user_validation=False): 
    print("************   BEGINNING FIND_EVENT PIPELINE   **************")
    print("Assuming start with the " + on_off_first + " observation.")
    
    #Opening list of files
    dat_file_list = open(dat_file_list_string).readlines()
    dat_file_list = [files.replace('\n','') for files in dat_file_list]
    dat_file_list = [files.replace(',','') for files in dat_file_list]
    n_files = len(dat_file_list)
    
    print("There are " + str(len(dat_file_list)) + " total files in your filelist, " + dat_file_list_string)
    print("Therefore, looking for events in " + str(int(n_files/number_in_sequence)) + " on-off sets")
    print("with a minimum SNR of " + str(SNR))
    
    if filter_level == 1:
        print("present in the A source with no RFI rejection from the off-sources")
    if filter_level == 2:
        print("Present in at least one A source with RFI rejection from the off-sources")
    if filter_level == 3:
        print("Present in all A sources with RFI rejection from the off-sources")
    
    if zero_drift_parameter == False:
        print("not including signals with zero drift")
    if zero_drift_parameter == True:
        print("including signals with zero drift")
    if saving == False:
        print("not saving the output files")
    if saving == True:
        print("saving the output files")    
    
    if user_validation == True:
        question = "Do you wish to proceed with these settings?"
        while "the answer is invalid":
            reply = str(input(question+' (y/n): ')).lower().strip()
            if reply[0] == 'y':
                break
            if reply[0] == 'n':
                return
    
    #Looping over n_files chunks.
    candidate_list = []
    for i in range(int(len(dat_file_list)/n_files)):
        file_sublist = dat_file_list[n_files*i:n_files*(i+1)]
        if on_off_first == 'ON':
            name=file_sublist[0].split('_')[5]  
        if on_off_first == 'OFF':
            name=file_sublist[1].split('_')[5]   
        print(name)
        cand = find_event.find_events(file_sublist, SNR_cut=SNR, check_zero_drift=zero_drift_parameter, filter_threshold=filter_level, on_off_first=on_off_first, number_in_sequence=number_in_sequence)
        
        if len(cand) > 0 or type(cand) != None:
            candidate_list.append(cand)
    if len(candidate_list) > 0:
        find_event_output_dataframe = pd.concat(candidate_list)
    else:
        "Sorry, no potential candidates with your given parameters :("
        find_event_output_dataframe = []

    print("ENDING PIPELINE")
    
    if saving == True:
        if zero_drift_parameter == True:
            filestring = name + '_f' + str(filter_level) + '_snr' + str(SNR) + '_zero' + '.csv'
        else:
            filestring = name + '_f' + str(filter_level) + '_snr' + str(SNR) + '.csv'
        
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


