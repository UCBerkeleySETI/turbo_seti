#!/usr/bin/env python

import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os
import matplotlib
import matplotlib.pylab as plt
from blimpy import Filterbank
from blimpy import Waterfall
import numpy as np
from blimpy.utils import db, lin, rebin, closest
import find_candidates

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.

MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot
MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot
MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory
MAX_HEADER_BLOCKS   = 100                    # Max size of header (in 512-byte blocks)


fontsize=18
font = {'family' : 'serif',
        'size'   : fontsize}

#------


def plot_waterfall(fil, f_start=None, f_stop=None, if_id=0, logged=True,cb=False,freq_label=False,MJD_time=False, **kwargs):
    """ Plot waterfall of data

    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """

    matplotlib.rc('font', **font)

    plot_f, plot_data = fil.grab_data(f_start, f_stop, if_id)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]

    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]

    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    if MJD_time:
        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])
    else:
        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    this_plot = plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        cmap='viridis_r',
        **kwargs
    )
    if cb:
        plt.colorbar()

    if freq_label:
        plt.xlabel("Frequency [Hz]",fontdict=font)
    if MJD_time:
        plt.ylabel("Time [MJD]",fontdict=font)
    else:
        plt.ylabel("Time [s]",fontdict=font)

    return this_plot

def make_waterfall_plots(filenames_list,target,f_start,f_stop,ion = False,epoch=None,local_host='',plot_name='',save_pdf_plot=False,saving_fig=False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''

    matplotlib.rc('font', **font)

    if ion:
        plt.ion()

    min_val = 0
    max_val = 5.
    factor = 1e6
    units = 'Hz'
    print target

    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    #finding plotting values range
    fil = Filterbank(filenames_list[0], f_start=f_start, f_stop=f_stop)
    plot_f, plot_data = fil.grab_data(f_start, f_stop, 0)
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    A1_avg = np.median(plot_data)
    A1_max = plot_data.max()
    A1_std = np.std(plot_data)

    if not epoch:
        epoch = fil.header['tstart']

    labeling = ['A','B','A','C','A','D']

#    delta_f = ('%f0.6'%np.abs(f_start-f_stop))
    delta_f = np.abs(f_start-f_stop)
    mid_f = np.abs(f_start+f_stop)/2.

    for i,filename in enumerate(filenames_list):
        print filename
        plt.subplot(n_plots,1,i+1)

        fil = Filterbank(filename, f_start=f_start, f_stop=f_stop)
        this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg-A1_std*min_val,vmax=A1_avg+max_val*A1_std,**kwargs)

        if i == 0:
            plt.title(target.replace('HIP','HIP '))

        if i < len(filenames_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #Some plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    print 'delta_f', delta_f
    plt.xticks(np.arange(f_start, f_stop, delta_f/4.),[round(loc_freq) for loc_freq in np.arange((f_start-mid_f), (f_stop-mid_f), delta_f/4.)*factor ])
    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)

    #to plot color bar. for now.
    cax = fig[0].add_axes([0.9, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Power [Arbitrary Units]')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    plt.subplots_adjust(hspace=0,wspace=0)

    if saving_fig:
        if not plot_name:
            plot_name = 'Candidate_waterfall_plots.%s.t%.0f.%s.f%.0f.png'%(target,epoch,local_host,mid_f*1e6)

        print 'Saving png figure.'
        plt.savefig(plot_name,bbox_inches='tight')
        if save_pdf_plot:
            print 'Saving pdf figure.'
            plt.savefig(plot_name.replace('.png','')+'.pdf', format='pdf', dpi=300,bbox_inches='tight')

def get_single_event_info(filename,freq_range = 0.001,make_latex_table=False,all=False):
    ''' This is in beta.
    '''

    #filename = full_path+ csv
    AAA_candidates = pd.read_csv(filename)

    if all:
        return AAA_candidates

    targets = list(AAA_candidates.groupby('Source').count().index)

    if len(targets) > 1:
        raise Error('Too many targets: ', targets)
    else:
        target = targets[0]

    #Taking only one hit per event !!!
    AAA_single = AAA_candidates[AAA_candidates['Source'] == target]
    AAA1_single = AAA_single[AAA_single['status'] == 'A1_table'].sort('SNR')

    f_start = AAA1_single['Freq'].values[-1] - freq_range
    f_stop = AAA1_single['Freq'].values[-1] + freq_range
    epoch = AAA1_single['MJD'].values[-1]

    if make_latex_table:
        # For making table of events
        for_table = [AAA1_single['Source'].values[0],'%.5f'%AAA1_single['Freq'].values[-1],'%.3f'%AAA1_single['DriftRate'].values[-1],'%.1f'%AAA1_single['SNR'].values[-1]]
        table_events+='  &  '.join(for_table)+'\ \ \n'

    return target,f_start,f_stop,epoch


if __name__ == "__main__":
    ''' Make it happen moment.
    '''

    raise Error('This is deprecated. Sorry ...')
#
#     #---------------------------
#     # Read in the full "A list" of stars
#     # This comes from the BL database.
#     #---------------------------
#     local_host = socket.gethostname()
#
#     if 'bl' in local_host:
#         dat_dit = '/datax/users/eenriquez/L_band_headquarters/hits_logistics/'
#
#     else:
#         dat_dit = '/Users/jeenriquez/RESEARCH/software/Lband_seti/analysis/'
#
#     make_latex_table = False
#
#     #---------------------------
#     AAA_candidates = pd.read_csv(dat_dit+'AAA_candidates.v4_1492476400.csv')
#     targets = list(AAA_candidates.groupby('Source').count().index)
#
#     table_events =''
#
#     for target in targets:
#         AAA_single = AAA_candidates[AAA_candidates['Source'] == target]
#         print target
#         filenames_list = get_filenames_list(target)
#
#         AAA1_single = AAA_single[AAA_single['status'] == 'A1_table'].sort('SNR')
#
#         f_start = AAA1_single['Freq'].values[-1] - 0.001
#         f_stop = AAA1_single['Freq'].values[-1] + 0.001
#         coarse_channel=AAA1_single['CoarseChanNum'].values[-1]
#
#         epoch = AAA1_single['MJD'].values[-1]
#
#         make_waterfall_plots(filenames_list,target,f_start,f_stop,ion=True,epoch=epoch,local_host=local_host)
#
#         if make_latex_table:
#             # For making table of events
#             for_table = [AAA1_single['Source'].values[0],'%.5f'%AAA1_single['Freq'].values[-1],'%.3f'%AAA1_single['DriftRate'].values[-1],'%.1f'%AAA1_single['SNR'].values[-1]]
#             table_events+='  &  '.join(for_table)+'\ \ \n'
#
#     stop
#
#
#     #Making table of events
#     with open('L_band_top_events.lst','w') as file_list:
#         file_list.write(table_events)
#
#     #Removing a bunch of RFI regions (GPS and so on).
#     AAA_candidates = remomve_RFI_regions(AAA_candidates)


