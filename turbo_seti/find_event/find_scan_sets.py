#!/usr/bin/env python
"""

This module is very BL-GBT oriented. Will transfer to different repo.

 ..author: Emilio Enriquez
"""

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pylab as plt
import socket
import numpy as np

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

def find_scan_sets(filename,band,ok_bands = ['L','S']):
    """Finding scan sets, needs filename of csv file and band.

    Args:
      filename: 
      band: 
      ok_bands:  (Default value = ['L')
      'S']: 

    Returns:

    """

    if '.csv' in filename:
        spider_danny = True
    else:
        spider_danny = False

    #---------------------------
    # Read in the full "A list" of stars
    # This comes from the BL database.
    #---------------------------
    local_host = socket.gethostname()

    #To Do:
    if 'bl' in local_host:
        master_file = open('/home/obs/logs/target_list_5-50pc.lst')
    else:
        master_file = open('/Users/jeenriquez/RESEARCH/SETI_BL/L_band/target_list_5-50pc.lst')

    a_list_master = master_file.read().splitlines()

    #---------------------------
    # Find the good  data by:
    # - Reading in the output from spider.
    #---------------------------
    # Initial data frame set up

    if spider_danny:
        try:
            df2 = pd.read_csv(filename) # New style spider files created by Danny's code.
            #,Unnamed: 0,az_start,data_type,fch1,filepath,filesize,foff,host,ibeam,machine_id,nbeams,nbits,nchans,nifs,nints,rawdatafile,source_name,src_dej,src_raj,telescope_id,tsamp,tstart
        except:
            IOError('Error opening file: %s'%filename)

        #column_names
        file = 'filepath'
        freq_chan1 = 'fch1'
        nchans = 'nchans'
        source_name = 'source_name'
        nints = 'nints'
        tstart = 'tstart'

        #Selection of high resolution data
        df3 = df2[df2[file].str.contains("gpuspec.0000.h5",na=False)]

    else:
        try:
            df = pd.read_csv(filename, sep=",|=", header=None,engine='python')    # Old style spider files.
        except:
            IOError('Error opening file: %s'%filename)

        #Reorganizing
        df2 = df.ix[:,1::2] #data
        df2.columns = list(df.ix[0,0::2]) #header

        #column_names
        file = 'file'
        freq_chan1 = 'Frequency of channel 1 (MHz)'
        nchans = 'Number of channels'
        source_name = 'Source Name'
        nints = 'Number of samples'
        tstart = 'Time stamp of first sample (MJD)'

        #Selection of high resolution data
        df3 = df2[df2[file].str.contains("gpuspec.0000.fil",na=False)]

    #Selection of observations from given band (soft)
    if band == 'S':
        df3 = df3[((df3[freq_chan1] > 2400.) & (df3[freq_chan1] < 3500.))]
    elif band == 'L':
        df3 = df3[df3[freq_chan1] < 2500]
    else:
        raise ValueError('Please probide one of the available bands:' + ok_bands)

    #---------------------------
    # Adding some extra columns for later look at the good set of data.

    df3['bands_used'] = [df3[file][ii].split('/')[-1].split('_')[1].replace('blc','') for ii in df3.index]
    df3['mid_Freq'] = df3[freq_chan1]-2.7939677238464355e-06*df3[nchans]/2.
    if band == 'L':
        df3['mid_Freq2'] = df3[freq_chan1]-2.835503418452676e-06*df3[nchans]/2.

    df3[source_name] = df3[source_name].str.upper()

    #---------------------------
    #Check the data that has the good bands.
    if band == 'S':
        df3 = df3[df3['bands_used'].str.contains('.1.2.3.4.5.6')]  # The dots are use as while cards (*)
    elif band == 'L':
        df3 = df3[df3['bands_used'].str.contains('.2.3.4.5')]
    else:
        raise ValueError('Please probide one of the available bands:' + ok_bands)

    #---------------------------
    #Check for correct Number of Samples
    df3 = df3[df3[nints] == 16]

    #---------------------------
    #Check for high resolution data with the bad central frequency.
    # The two main frequency resolutions used are -2.7939677238464355e-06 and -2.835503418452676e-06  .

    if band == 'L':
        # Ok mid Freqs around 1501.4648, or 1475.0976

        df_good_mid_FreqA = df3[((df3['mid_Freq'] > 1501.4) & (df3['mid_Freq'] < 1501.5))]
        df_good_mid_FreqB = df3[((df3['mid_Freq2'] > 1501.4) & (df3['mid_Freq2'] < 1501.5))]

        df_good_mid_FreqA2 = df3[((df3['mid_Freq'] > 1475.0) & (df3['mid_Freq'] < 1475.1))]
        df_good_mid_FreqB2 = df3[((df3['mid_Freq2'] > 1475.0) & (df3['mid_Freq2'] < 1475.1))]

        #bad mid freqs?

        #concat
        df3 = pd.concat([df_good_mid_FreqA,df_good_mid_FreqA2,df_good_mid_FreqB,df_good_mid_FreqB2])

    elif band == 'S':
        # Ok mid Freqs around 2300.3906, 2269.6289, 2307.75
        df3 = df3[((df3['mid_Freq'] > 2269.) & (df3['mid_Freq'] < 2310.))]
        #bad mid freqs 2276.23
        df3 = df3[~((df3['mid_Freq'] > 2276.2) & (df3['mid_Freq'] < 2276.3))]
    else:
        raise ValueError('Please probide one of the available bands:' + ok_bands)

    #---------------------------
    # Apply format change in tstart and sort by it.

    df3[tstart] = df3[tstart].apply(pd.to_numeric)
    df3 = df3.sort_values(by=[tstart])

    #---------------------------
    # Selecting only the targets in the A-list

    #Selecting all the targets from the B list
    df_targets_blist = df3[~df3[source_name].isin(a_list_master)]
    df_targets_clist =  df3[df3[source_name].str.contains('_OFF',na=False)]
    df_targets_blist = pd.concat([df_targets_blist,df_targets_clist])
    else_list = df_targets_blist[source_name].unique()

    #Selecting all the good targets from the A list
    df_targets_alist = df3[~df3[source_name].isin(else_list)]

    #---------------------------
    #Showing some info

    print('------      o      --------')
    a_unique = df_targets_alist[source_name].unique()
    print('This list was created for the %s band data'%band)
    print('The total number of targets from the A-list that:')
    print('Observed and spliced is      : %i'%(len(a_unique)))

    #---------------------------
    # Group the df_targets and look for the ones observed 3 times or more
    # Grouping without date constrains.
    df_targets = df_targets_alist.groupby(source_name).count()[file] > 2
    df_bool_list = df_targets.tolist()
    list_completed = list(df_targets[df_bool_list].index.values)

    #---------------------------
    #Selecting targets with "completed" observations
    df_targets_alist = df_targets_alist[df_targets_alist[source_name].isin(list_completed)]
    alist_completed_unique = df_targets_alist[source_name].unique()

    print('Have at least 3 observations : %i'%(len(alist_completed_unique)))

    #---------------------------
    #Creating list of targets

    list_targets =''
    list_A_stars=[]
    list_A_times=[]

    counting = 0



    for a_star in alist_completed_unique:

        df_a_star = df_targets_alist[df_targets_alist[source_name] == a_star]

#         if len(df_a_star[tstart]) > 5:
#             print 'WARNING: Too many observations of target: ', a_star
#
#         mid_time = df_a_star[tstart].median()
#         df_a_star['delta_t'] = df_a_star[tstart].apply(lambda x: float(x) - float(mid_time))
#
#         #Taking observations only from the same day.
#         df_a_star = df_a_star[df_a_star['delta_t'] < 1.]

        list_a_star_times = df_a_star[tstart].unique()
#        scan_count = 0

        for a_time in list_a_star_times:

            #Calculating delta t
            df_a_star['delta_t'] = df_a_star[tstart].apply(lambda x: np.abs(float(x) - float(a_time)))
            time_list = df_a_star[df_a_star['delta_t'] < 0.05][tstart]

            if  len(time_list)>3:

                if np.isin(time_list,list_A_times).sum() > 2:
                    print('WARNING: Skiping this A observation:', a_star,a_time, 'Length is :', len(time_list))
                    continue

            #Taking observations only with 3 nearby.
#             This fails when having both many observations of one target, but some
#             if len(time_list)>3:
#                 scan_count+=1
#                 if scan_count > 3:
#                     scan_count = 0
#                     print 'WARNING: Skiping this A observation:', a_star,a_time, 'Length is :', len(time_list)
#                     continue
#
# #                 mid_time = time_list.median()
# #                 time_diffs = np.abs((time_list- mid_time).unique())
# #
# #                 if np.abs(a_time - mid_time) > time_diffs[2]+.0001:
# #                     print 'WARNING: Skiping this A observation:', a_star,a_time, 'Length is :', len(time_list)
# #                     continue


            elif  len(time_list)<3:
                continue

            #Making tmp DF for ONs and OFFs
            df_tmp = df3[ (df3[tstart] > float(a_time)-0.1) & (df3[tstart] < float(a_time)+0.1)]
            df_tmp['delta_t'] = df_tmp[tstart].apply(lambda x: float(x) - float(a_time))

            try:
                ii = df_tmp[df_tmp['delta_t']>0.001]['delta_t'].idxmin()   #Find B star index  #.001 = 1.44 min
                #jj = df_tmp[df_tmp['delta_t']>=0]['delta_t'].idxmin()   #Find A star index

                #full path
                a_name = list(df_a_star[df_a_star[tstart] == a_time][file])[0]
                b_name = df_tmp[file][ii]

            except:
                print('WARNING: Could not find ON-OFF pair: ', a_star, a_name)

                #full path
                a_name = list(df_a_star[df_a_star[tstart] == a_time][file])[0]
                b_name = 'Place_holder_to_delete.'   # Creates a place holder in the list of obs for manual deletion.

#                continue

            if a_name == b_name:
                print('WARNING: Skipping (a=b). ', a_name)
#                continue

            #Find if data pairs are not in the same node (thus 'non-co-living').
#            if a_name.split('/')[1] != b_name.split('/')[1]:
#                print 'WARNING: A and B not in same location.', a_name
#                 continue
#             else:
            #a_star_file_name, b_star_file_name
            tmp_string = ['/mnt_'+local_host+a_name,'\n','/mnt_'+local_host+b_name]
            list_targets += ''.join(tmp_string)+'\n'
            counting+=1

            list_A_times.append(a_time)



        list_A_stars.append(a_star+'\n')

    print('Actual number of stars at least one complete set: %i'%(len(list_A_stars)))

    #---------------------------
    #Save lists

    with open('%s_band_target_pairs.lst'%band,'w') as file_list:
    #with open('/datax/users/eenriquez/L-band_analysis/'+'%s_band_target_pairs.lst'%band,'w') as file_list:
        file_list.write(list_targets)

    with open('%s_band_A_stars.lst'%band,'w') as file_list:
    #with open('/datax/users/eenriquez/L-band_analysis/'+'%s_band_A_stars.lst'%band,'w') as file_list:
        file_list.write(''.join(list_A_stars))

def main():
    """Main funtion for find_event scripts."""

    parser = ArgumentParser(description="Command line utility for creating a list of ON/OFF pairs of spliced data.")
    parser.add_argument('filename', type=str, help='Full path and filename to read (csv).')
    parser.add_argument('band', type=str,default='L', help='Which band to use? (L,S)')
    args = parser.parse_args()

    #Available bands
    ok_bands = ['L','S']

    if args.band not in ok_bands:
        raise ValueError('Please probide one of the available bands:' + ok_bands)
    else:
        band = args.band

    filename = args.filename

    find_scan_sets(filename,band,ok_band=ok_bands)

if __name__ == "__main__":
    main()

