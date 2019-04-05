#!/usr/bin/env python
''' ..author: Emilio Enriquez
'''

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pylab as plt
import socket

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

parser = ArgumentParser(description="Command line utility for creating a list of On/OFF pairs of spliced data.")
parser.add_argument('filename', type=str, help='Full path and filenmae to read.')
parser.add_argument('band', type=str,default='L', help='Which band to use?')
args = parser.parse_args()

#Available bands
ok_bands = 'L'

if args.band not in ok_bands:
    raise ValueError('Please probide one of the available bands:' + ok_bands)
else:
    band = args.band

filename = args.filename

#---------------------------
# Read in the full "A list" of stars
# This comes from the BL database.
#---------------------------
local_host = socket.gethostname()

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



else:

try:
    #df = pd.read_csv(filename, sep=",|=", header=None,engine='python')    # Old style spider files.
    df = pd.read_csv(filename) # New style spider files created by Danny's code.

except:
    IOError('Error opening file: %s'%filename)


#,Unnamed: 0,az_start,data_type,fch1,filepath,filesize,foff,host,ibeam,machine_id,nbeams,nbits,nchans,nifs,nints,rawdatafile,source_name,src_dej,src_raj,telescope_id,tsamp,tstart





df2 = df.ix[:,1::2]
df2.columns = list(df.ix[0,0::2])

#Selection of high resolution data
df3 = df2[df2['file'].str.contains("gpuspec.0000.fil",na=False)]

#Selection of observations from given band (soft)
if band == 'S':
    df3 = df3[((df3['Frequency of channel 1 (MHz)'] > 2400.) & (df3['Frequency of channel 1 (MHz)'] < 3500.))]
elif band == 'L':
    df3 = df3[df3['Frequency of channel 1 (MHz)'] < 2500]
else:
    raise ValueError('Please probide one of the available bands:' + ok_bands)

#---------------------------
# Adding some extra columns for later look at the good set of data.

df3['bands_used'] = [df3['file'][ii].split('/')[-1].split('_')[1].replace('blc','') for ii in df3.index]
df3['mid_Freq'] = df3['Frequency of channel 1 (MHz)']-2.835503418452676e-06*df3['Number of channels']/2.
df3['mid_Freq2'] = df3['Frequency of channel 1 (MHz)']-2.7939677238464355e-06*df3['Number of channels']/2.

df3['Source Name'] = df3['Source Name'].str.upper()

#---------------------------
#Check the data that has the 4 good bands (02030405).
if band == 'S':
    df3 = df3[df3['bands_used'].str.contains('0001020304050607')]
elif band == 'L':
    df3 = df3[df3['bands_used'].str.contains('02030405')]
else:
    raise ValueError('Please probide one of the available bands:' + ok_bands)

#---------------------------
#Check for correct Number of Samples
df3 = df3[df3['Number of samples'] == 16 ]

#---------------------------
#Check for high resolution data with the bad central frequency.
# The two main frequency resolutions used are -2.7939677238464355e-06 and -2.835503418452676e-06  .

if band == 'L':
    df_good_mid_Freq = df3[((df3['mid_Freq'] > 1501.4) & (df3['mid_Freq'] < 1501.5))]
    df_good_mid_Freq2 = df3[((df3['mid_Freq2'] > 1501.4) & (df3['mid_Freq2'] < 1501.5))]
else:
    raise ValueError('Please probide one of the available bands:' + ok_bands)

df3 = pd.concat([df_good_mid_Freq,df_good_mid_Freq2])

#---------------------------
# Apply format change in

df3['Time stamp of first sample (MJD)'] = df3['Time stamp of first sample (MJD)'].apply(pd.to_numeric)

#---------------------------
# Selecting only the targets in the A-list

#Selecting all the targets from the B list
df_targets_blist = df3[~df3['Source Name'].isin(a_list_master)]
df_targets_clist =  df3[df3['Source Name'].str.contains('_OFF',na=False)]
df_targets_blist = pd.concat([df_targets_blist,df_targets_clist])
else_list = df_targets_blist['Source Name'].unique()

#Selecting all the good targets from the A list
df_targets_alist = df3[~df3['Source Name'].isin(else_list)]

#---------------------------
#Showing some info

print '------      o      --------'
a_unique = df_targets_alist['Source Name'].unique()
print 'This list was created for the %s band data'%band
print 'The total number of targets from the A-list that:'
print 'Observed and spliced is      : %i'%(len(a_unique))

#---------------------------
#Group the df_targets and look for the ones observed 3 times or more
# Grouping without date constrains.
df_targets = df_targets_alist.groupby('Source Name').count()['file'] > 2
df_bool_list = df_targets.tolist()
list_completed = list(df_targets[df_bool_list].index.values)

#---------------------------
#Selecting targets with "completed" observations
df_targets_alist = df_targets_alist[df_targets_alist['Source Name'].isin(list_completed)]
alist_completed_unique = df_targets_alist['Source Name'].unique()

print 'Have at least 3 observations : %i'%(len(alist_completed_unique))

#---------------------------
#Creating list of targets

list_targets =''
list_A_stars=''
i = 0

for a_star in alist_completed_unique:

    df_a_star = df_targets_alist[df_targets_alist['Source Name'] == a_star]
    list_a_star_times = df_a_star['Time stamp of first sample (MJD)'].unique()

    for a_time in list_a_star_times:

        df_tmp = df3[ (df3['Time stamp of first sample (MJD)'] > float(a_time)-0.1) & (df3['Time stamp of first sample (MJD)'] < float(a_time)+0.1)]
        df_tmp['delta_t'] = df_tmp['Time stamp of first sample (MJD)'].apply(lambda x: float(x) - float(a_time))

        try:
            ii = df_tmp[df_tmp['delta_t']>0.001]['delta_t'].idxmin()   #Find B star index  #.001 = 1.44 min
            jj = df_tmp[df_tmp['delta_t']>=0]['delta_t'].idxmin()   #Find A star index
        except:
            continue

#             a_name = '/datax'+list(df_a_star[df_a_star['Time stamp of first sample (MJD)'] == a_time]['file'])[0].split('datax')[-1].replace('0000.fil','0002.fil')
#             b_name = '/datax'+df_tmp['file'][ii].split('datax')[-1].replace('0000.fil','0002.fil')
        #Only name
#             a_name = 'spliced'+list(df_a_star[df_a_star['Time stamp of first sample (MJD)'] == a_time]['file'])[0].split('spliced')[-1]
#             b_name = 'spliced'+df_tmp['file'][ii].split('spliced')[-1]

        #full path
        a_name = list(df_a_star[df_a_star['Time stamp of first sample (MJD)'] == a_time]['file'])[0]
        b_name = df_tmp['file'][ii]

        if a_name == b_name:
            print 'WARNING: Skiping (a=b). ',node, a_name
            continue

        #Find if data pairs are not in the same node (thus 'non-co-living').
        if a_name.split('/')[1] != b_name.split('/')[1]:
            print 'WARNING: skiping since A and B not in same location.',node, a_name
            continue
        else:
            #a_star_file_name, b_star_file_name
            tmp_string = ['/mnt_'+local_host+a_name,'\n','/mnt_'+local_host+b_name]
            list_targets += ''.join(tmp_string)+'\n'
            i+=1

    list_A_stars+=a_star+'\n'

#---------------------------
#Save lists

with open('%s_band_target_pairs.lst'%band,'w') as file_list:
#with open('/datax/users/eenriquez/L-band_analysis/'+'%s_band_target_pairs.lst'%band,'w') as file_list:
    file_list.write(list_targets)


with open('%s_band_A_stars.lst'%band,'w') as file_list:
#with open('/datax/users/eenriquez/L-band_analysis/'+'%s_band_A_stars.lst'%band,'w') as file_list:
    file_list.write(list_A_stars)



