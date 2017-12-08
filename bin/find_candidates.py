#!/usr/bin/env python
'''
Script to find candidates in a group of ABACAD observations.
    1) It compares ON vs OFF observations
    2) any signal found only in the ON is checked to be found in all the 3 ONs following the frequency drift of the signal.

Usage (beta):
    import find_candidates_test as fact
    TEST_LIST = ['dir/filename_A1.dat','dir/filename_B1.dat','dir/filename_A2.dat','dir/filename_B2.dat','dir/filename_A3.dat','dir/filename_B3.dat']
    fact.find_candidates(TEST_LIST)

    ..author: Emilio Enriquez (jeenriquez@gmail.com)
'''

import pandas as pd
import pdb;# pdb.set_trace()
import time
from blimpy import Filterbank
import numpy as np
from blimpy.utils import db, lin, rebin, closest
from optparse import OptionParser
import plot_candidates
import sys

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.
#------

def make_table(filename,init=False):
    ''' This creates a pandas dataframe from a turboSETI flat file.
    '''

    if init:
        columns = ['FileID','Source','MJD','RA','DEC','TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 'FreqStart', 'FreqEnd', 'CoarseChanNum', 'FullNumHitsInRange','status','Hit_ID','ON_in_range','RFI_in_range']
        df_data = pd.DataFrame(columns=columns)

    else:

        file_dat = open(filename)
        hits = file_dat.readlines()

        #Info from header
        FileID = hits[1].strip().split(':')[-1].strip()
        Source = hits[3].strip().split(':')[-1].strip()

        MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
        RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
        DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

        #Info from individual Hits
        all_hits = [hit.strip().split('\t') for hit in hits[9:]]

        TopHitNum = zip(*all_hits)[0]
        DriftRate = zip(*all_hits)[1]
        SNR = zip(*all_hits)[2]
        Freq = zip(*all_hits)[3]
        ChanIndx = zip(*all_hits)[5]
        FreqStart = zip(*all_hits)[6]
        FreqEnd = zip(*all_hits)[7]
        CoarseChanNum = zip(*all_hits)[10]
        FullNumHitsInRange = zip(*all_hits)[11]

        data = {'TopHitNum':TopHitNum,
                'DriftRate':[float(df) for df in DriftRate],
                'SNR':[float(ss) for ss in SNR],
                'Freq':[float(ff) for ff in Freq],
                'ChanIndx':ChanIndx,
                'FreqStart':FreqStart,
                'FreqEnd':FreqEnd,
                'CoarseChanNum':CoarseChanNum,
                'FullNumHitsInRange':FullNumHitsInRange
                }

        df_data = pd.DataFrame(data)
        df_data = df_data.apply(pd.to_numeric)

        #Adding header information.
        df_data['FileID'] = FileID
        df_data['Source'] = Source.upper()
        df_data['MJD'] = MJD
        df_data['RA'] = RA
        df_data['DEC'] = DEC

        #Adding extra columns.
        df_data['Hit_ID'] = ''
        df_data['status'] = ''
        df_data['ON_in_range'] = ''
        df_data['RFI_in_range'] = ''

    return df_data

def calc_freq_range(hit,delta_t=0,max_dr=True,follow=False):
    '''Calculates the range of frequencies given a freq and drift_rate.
    '''

    if max_dr:
        drift_rate = MAX_DRIFT_RATE
    else:
        drift_rate = hit['DriftRate']

    if follow:
        freq = hit['Freq'] + drift_rate*(delta_t)/1000.
        delta_t = 2*OBS_LENGHT  # NOTE: I'm doing twice the amount. To widen the range a bit. Still it is dependend on hit['DriftRate']
    else:
        freq = hit['Freq']
        delta_t = delta_t+OBS_LENGHT  #adding to the end of the obs.

    low_bound  = freq - abs(drift_rate)*delta_t/1000.
    high_bound = freq + abs(drift_rate)*delta_t/1000.

    return [low_bound,high_bound]

def follow_candidate(hit,A_table,get_count=True):
    ''' Follows hit to another obs, and finds if antithing there.
    '''

    freq_range = calc_freq_range(hit,delta_t=A_table['delta_t'].values[0],max_dr=False,follow=True)
    new_A_table = A_table[(A_table['Freq']>freq_range[0]) & (A_table['Freq']<freq_range[1])]

    if get_count:
        n_hits_in_range = len(new_A_table)

        #Avoiding cases where multiple hits in one obs, and none in the other.
        if n_hits_in_range:
            return 1
        else:
            return 0
    else:
        return new_A_table

def search_hits(A_table_list,B_table, SNR_cut = 15, check_zero_drift=False):
    '''Rejects hits based on some logic.
    '''

    A_table = pd.concat(A_table_list)

    #Removing non-drift signals
    if check_zero_drift:
        And0_table = A_table
    else:
        And0_table = A_table[A_table['DriftRate'] != 0.0]

    #Make the SNR_cut
    Asc_table = And0_table[And0_table['SNR']> SNR_cut]
    if not (len(Asc_table) > 0):
        print 'NOTE: Found no hits above the SNR cut.'
        return Asc_table,0
    else:
        print 'NOTE: Found hits above the SNR cut.'

    # Finding RFI within a freq range by comparing ON to OFF obs.
    Asc_table['RFI_in_range'] = Asc_table.apply(lambda hit: len(B_table[((B_table['Freq'] > calc_freq_range(hit)[0]) & (B_table['Freq'] < calc_freq_range(hit)[1]))]),axis=1)
    AnB_table = Asc_table[Asc_table['RFI_in_range'] == 0]
    if not (len(AnB_table) > 2):
        print 'NOTE: Found no hits present only on the A observations.'
        return AnB_table, 1
    else:
        print 'NOTE: Found hits present only on the A observations.'

    #Find the ones that are present in all the 3 ON obs, and follow the drifted signal.
    A1nB_table = AnB_table[AnB_table['status'] == 'A1_table']
    A2nB_table = AnB_table[AnB_table['status'] == 'A2_table']
    A3nB_table = AnB_table[AnB_table['status'] == 'A3_table']
    if len(A1nB_table) > 0  and len(A2nB_table) > 0 and len(A3nB_table) > 0:
        A1nB_table['ON_in_range'] = A1nB_table.apply(lambda hit: follow_candidate(hit,A2nB_table) + follow_candidate(hit,A3nB_table) ,axis=1)
        AA_table = A1nB_table[A1nB_table['ON_in_range'] == 2]
    else:
        print 'NOTE: Found no hits present in all three A observations.'
        return AnB_table,2

    if len(AA_table) > 0:
        print 'NOTE: Found some candidates! :)'
    else:
        print 'NOTE: Found no candidates. :('
        return AA_table,2

    #Create list of candidates.
    AAA_table_list = []

    for hit_index, hit in AA_table.iterrows():
        A1i_table = follow_candidate(hit,A1nB_table,get_count=False)
        A1i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
        A2i_table = follow_candidate(hit,A2nB_table,get_count=False)
        A2i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
        A3i_table = follow_candidate(hit,A3nB_table,get_count=False)
        A3i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)

        AAA_table_list += [A1i_table, A2i_table, A3i_table]

    AAA_table = pd.concat(AAA_table_list)

    return AAA_table,3

def find_candidates(file_list,SNR_cut=10,check_zero_drift=False,flag_threshold=3):
    ''' Reads a list of flat turboSETI files, the list should be in the ABACAD configuration.
        It calls other functions to find candidates within this group of files.
        Flag_threshold allows the return of a table of candidates with hits at different levels of filtering.
        Flag_threshold = [1,2,3] means [ hits above an SNR cut witout AB check, only in some As, only in all As]
    '''
    #---------------------------
    #Setting things up.

    t0 = time.time()

    #---------------------------
    #Looping over list

    A_table_list = []
    B_table_list = []

    for i,flat_file in enumerate(file_list):
        kk = 1   # counter for table name.
        ll = 1   # counter for table name.

        #---------------------------
        # Reading hits data
        print 'Reading hits data for %s.'%flat_file

        # Checking if on A or B observation.
        if i%2:

            Bi_table=make_table(flat_file)
            Bi_table['status'] = 'B%i_table'%ll
            print 'Ther are %i hits on this file.'%len(Bi_table)

            #---------------------------
            #Grouping all hits per obs set.
            B_table_list.append(Bi_table)

            ll+=1

        else:
            Ai_table=make_table(flat_file)
            Ai_table['status'] = 'A%i_table'%kk
            print 'Ther are %i hits on this file.'%len(Ai_table)

            #---------------------------
            #Grouping all hits per obs set.
            A_table_list.append(Ai_table)

            kk+=1

    #Concatenating
    A_table = pd.concat(A_table_list)
    B_table = pd.concat(B_table_list)

    #To save all the hits. Uncomment these 3 lines.
#             all_candidates_list.append(A_table)  This blows up the mem. Caution.
#             all_candidates_list.append(B_table)
#            continue

    print 'Finding all candidates for this A-B set.'
    AAA_table,flag_level = search_hits(A_table_list,B_table,SNR_cut=SNR_cut,check_zero_drift=check_zero_drift)

    if len(AAA_table) > 0:
        print 'Found: %2.2f'%(len(AAA_table)/3.)

    print '------   o   -------'

    t1 = time.time()
    print 'Search time: %.2f sec' % ((t1-t0))

    if flag_level < flag_threshold:
        return None
    else:
        return AAA_table


#     Some snippets for possible upgrades of the code.
#     stop
#
#     #Concatenating all the candidates.
#     AAA_candidates = pd.concat(AAA_candidates_list,ignore_index=True)
#     AAA_candidates_list = 0.
#
#     #Save hits.
#     AAA_candidates.to_csv('AAA_candidates.v4_%.0f.csv'%time.time())
#
#     #Looking at some stats.
#     plt.ion()
#     plt.figure()
#     AAA_candidates['Freq'].plot.hist(bins=100,logy=True)#
#     plt.savefig('Frequency_hist.png')
#
#     plt.figure()
#     AAA_candidates['DriftRate'].plot.hist(bins=25,logy=True)
#
#     #Removing a bunch of RFI regions (GPS and so on).
#     AAA_candidates = remomve_RFI_regions(AAA_candidates)





def main():
    """ Main funtion for find_candidate scripts. """

    p = OptionParser()
    p.set_usage('python find_candidates.py [options]')
#    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='', help='Location for output files. Default: local dir. ')
    p.add_option('-n', '--number_files', dest='n_files', type='int', default=6, help='Number of files to check for candidates, standard is 6, for an ABACAD config.')
    p.add_option('-l', '--list', dest='file_list', type='str', default='out_dats.lst', help='List of files to run (without the path).')
    p.add_option('-p', '--plotting', dest='plotting', action='store_false', default=True, help='Boolean for plotting. Default True, use for False.')
    p.add_option('-s', '--saving', dest='saving', action='store_true', default=False, help='Boolean for saving plot and csv data. Default False, use for True.')
    p.add_option('-r', '--SNR_cut', dest='SNR_cut', type='int', default=10, help='SNR cut, default SNR=10.')
    p.add_option('-z', '--check_zero_drift', dest='check_zero_drift', action='store_true', default=False, help='Boolean for not ignoring zero drift hits, if True it will search them if only present in the ON. Default False, use for True.')
    p.add_option('-f', '--flag_threshold', dest='flag_threshold', type='int', default=3, help='Flag_threshold allows the return of a table of candidates with hits at different levels of filtering.')

    opts, args = p.parse_args(sys.argv[1:])

#    out_dir = opts.out_dir
    n_files = opts.n_files
    file_list = opts.file_list
    plotting = opts.plotting
    saving = opts.saving
    SNR_cut = opts.SNR_cut
    check_zero_drift = opts.check_zero_drift

    #---------------------

    #Opening list of files
    file_list = open(file_list).readlines()
    file_list = [files.replace('\n','') for files in file_list]

    #Check
    if len(file_list) < n_files:
        print "It seems len(file_list) < n_files assuming len(file_list) = n_files."
        n_files = len(file_list)

    #---------------------
    #Finding candidates.

    #Looping over n_files chunks.
    for i in range(len(file_list)/n_files):

        file_sublist = file_list[n_files*i:n_files*(i+1)]

        candidates = find_candidates(file_sublist,SNR_cut=SNR_cut,check_zero_drift=check_zero_drift,flag_threshold=flag_threshold)

        if len(candidates) and plotting:

            filenames = open('../processed_targets.lst').readlines()
            filenames = [files.replace('\n','') for files in filenames]

            candidate_index = candidates['FreqEnd'].index.unique()

            #Plotting individual candidates
            for ii in candidate_index:

                Freq_Start = candidates['FreqStart'][ii].unique()[0]
                Freq_End = candidates['FreqEnd'][ii].unique()[0]

                plot_candidates.make_waterfall_plots(filenames[n_files*i:n_files*(i+1)],candidates['Source'].unique()[0],Freq_Start,Freq_End,ion=True,save_pdf_plot=saving,saving_fig=saving)

        #Saving csv
        if saving:
            candidates.to_csv('Candidates_%.0f.csv'%time.time())


if __name__ == "__main__":
    main()
