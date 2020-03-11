#!/usr/bin/env python

import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os, sys
import matplotlib
import matplotlib.pylab as plt
from blimpy import Waterfall
import numpy as np
from blimpy.utils import db, lin, rebin, closest
from optparse import OptionParser


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

def make_waterfall_plots(filenames_list,f_start,f_stop,plot_range=True,target='',ion = False,epoch=None,local_host='',plot_name='',save_pdf_plot=False,saving_fig=False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)

        plot_range:  selecting vmin vmax from first On observation, or not.

    '''

    matplotlib.rc('font', **font)

    if ion:
        plt.ion()

    min_val = 0
    max_val = 5.
    factor = 1e6
    units = 'Hz'

    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    #finding plotting values range
    fil = Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)
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
        epoch = fil.header[b'tstart']

    if not target:
        target = fil.header[b'source_name']

    labeling = ['A','B','A','C','A','D']

#    delta_f = ('%f0.6'%np.abs(f_start-f_stop))
    delta_f = f_start-f_stop
    mid_f = np.abs(f_start+f_stop)/2.

    for i,filename in enumerate(filenames_list):
        print(filename)
        plt.subplot(n_plots,1,i+1)

        fil = Waterfall(filename, f_start=f_start, f_stop=f_stop)
        if plot_range:
            this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg-A1_std*min_val,vmax=A1_avg+max_val*A1_std,**kwargs)
        else:
            this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,**kwargs)

        if i == 0:
            plt.title(target.replace('HIP','HIP '))

        if i < len(filenames_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #Some plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    print('delta_f', delta_f)
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

        print('Saving png figure.')
        plt.savefig(plot_name,bbox_inches='tight')
        if save_pdf_plot:
            print('Saving pdf figure.')
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



def main():
    """ Main funtion for find_event scripts. """

    p = OptionParser()
    p.set_usage('python plot_events.py [options]')
#    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='', help='Location for output files. Default: local dir. ')
    p.add_option('-n', '--number_files', dest='n_files', type='int', default=6, help='Number of files to check for candidates, standard is 6, for an ABACAD config.')
    p.add_option('-L', '--list_fils', dest='fil_file_list', type='str', default=None, help='List of .fil files to run (with the path).')
    p.add_option('-p', '--plotting', dest='plotting', action='store_true', default=False, help='Boolean for plotting. Default False, use for True.')
    p.add_option('-s', '--saving', dest='saving', action='store_false', default=True, help='Boolean for saving plot in png and pdf. Default True, use for False.')
    p.add_option('-r', '--plot_range', dest='plot_range', action='store_false', default=True, help='Boolean for ploting vmin vmax with respect to first ON. Default True, use for False.')
    p.add_option('-b', '--f_start', dest='f_start', type='float', default=None, help='Start frequency (begin), in MHz')
    p.add_option('-e', '--f_stop', dest='f_stop', type='float', default=None, help='Stop frequency (end), in MHz')

    opts, args = p.parse_args(sys.argv[1:])

#    out_dir = opts.out_dir
    n_files = opts.n_files
    fil_file_list = opts.fil_file_list
    plotting = opts.plotting
    saving = opts.saving
    plot_range = opts.plot_range
    f_start = opts.f_start
    f_stop =  opts.f_stop

    if not fil_file_list:
        raise ValueError('Need to provide filename of list of .fil files.')

    if not f_start or not f_stop:
        raise ValueError('Need to provide f_start and f_stop')

    #---------------------

    local_host = socket.gethostname()

    #Opening list of files
    fil_file_list = open(fil_file_list).readlines()
    fil_file_list = [files.strip() for files in fil_file_list]

    #Check number of files matches
    if len(fil_file_list) < n_files:
        print("It seems len(fil_file_list) < n_files assuming len(fil_file_list) = n_files.")
        n_files = len(fil_file_list)

    #---------------------
    #plottign event.

    make_waterfall_plots(fil_file_list,f_start,f_stop,plot_range = plot_range, ion = plotting, local_host = local_host, save_pdf_plot = saving, saving_fig = saving)


if __name__ == "__main__":
    main()


