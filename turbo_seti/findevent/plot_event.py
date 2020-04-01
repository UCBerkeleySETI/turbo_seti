#!/usr/bin/env python
'''
Script to plot events
    ..author: Sofia Sheikh (ssheikhmsa@gmail.com)
'''

#General packages import
import glob
import os
import sys
import pandas as pd
#import pylab as plt
import numpy as np
import logging; logging.disable(logging.CRITICAL);

#BL + my packages import
import updated_find_event
from blimpy import Filterbank
from blimpy.filterbank import *
import blimpy as bl

#Plotting packages import
import matplotlib
import matplotlib.pyplot as plt

#preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)

def make_waterfall_plots(filenames_list, target, drates, fvals, f_start,f_stop, node_string, filter_level, ion=False,epoch=None,bw=250.0, local_host='',plot_name='',save_pdf_plot=False,saving_fig=False,offset=0,dedoppler=False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''
    
    #prepares for plotting
    print('Preparing to plot: ', target)
    matplotlib.rc('font', **font)
    if ion:
        plt.ion()

    #defines a minimum and maximum of... something
    min_val = 0
    max_val = 5.
    factor = 1e6
    units = 'Hz'

    #sets up the sub-plots
    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    #finding plotting values range for the first panel (A1)
    fil = bl.Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)
    t0 = fil.header['tstart']
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)
    dec_fac_x, dec_fac_y = 1, 1

    #rebinning data to plot correctly with fewer plots
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0] 
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    #investigate intensity values for A1 (first panel)
    plot_data = 10*np.log10(plot_data)
    A1_avg = np.median(plot_data)
    A1_max = plot_data.max()
    A1_std = np.std(plot_data)
    
    #defining more plot parameters
    delta_f = 0.000250
    epoch = fil.header['tstart']
    mid_f = np.abs(f_start+f_stop)/2.
    drate_max = np.max(np.abs(drates))
    
    subplots = []
    
    #working out intensity scale
    if kwargs.get('clim', None) is None:
        vmin=A1_avg-A1_std*min_val-2
        vmax=A1_avg+max_val*A1_std
    else:
        vmin, vmax = kwargs['clim']
        
    #Filling in each subplot for the full plot
    for i,filename in enumerate(filenames_list):
        subplot = plt.subplot(n_plots,1,i+1)
        subplots.append(subplot)
        fil = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop)
        try:
            this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop, drate=drate_max,
                                       vmin=vmin,vmax=vmax,**kwargs)
            for drate, fval in zip(drates, fvals):
                t_elapsed = Time(fil.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
                t_duration = (fil.n_ints_in_file -1)* fil.header['tsamp']
                f_event = fval + drate / 1e6 * t_elapsed
                overlay_drift(f_event, drate, t_duration, offset)
        except:
            raise
            
    #Titling the plot
        if i == 0:
            srcname = "%s $\dot{\\nu}$=%2.3f Hzs$^{-1}$" % (target, drate_max)
            plt.title(srcname)
    #Plot formatting
        if i < len(filenames_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #More plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    xloc = np.linspace(f_start, f_stop, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
    if np.max(xticks) > 1000:
        xticks = [xt/1000 for xt in xticks]
        units = 'kHz'
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)

    #Colorbar
    cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Power [dB counts]')
    
    #Adjust plots
    plt.subplots_adjust(hspace=0,wspace=0)
    
    #save the figures
    plt.savefig(node_string + '_f' + str(filter_level) + '_' + target[0] + '_dr_' + "{:0.2f}".format(drate_max) + '_freq_' "{:0.2f}".format(f_start) + ".png",
               bbox_inches='tight')

    return subplots

def overlay_drift(f_event, drate, t_duration, offset=0):
    if offset == 'auto':
        offset = - 0.2 * drate*t_duration
    plt.plot((f_event+offset/1e6, f_event+drate/1e6*t_duration+offset/1e6), (0, t_duration), c='#cc0000', ls='dashed', lw=2)
    
def plot_waterfall(fil, f_start=None, f_stop=None, drate=None, if_id=0, logged=True,cb=False,freq_label=False,MJD_time=False, **kwargs):
    """ Plot waterfall of data
    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """

    #prepare font
    matplotlib.rc('font', **font)

    #Get the data
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)

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

    #plots and scales intensity
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    kwargs['logged'] = True
    if kwargs['logged'] == True:
        plot_data = 10*np.log10(plot_data)
        kwargs.pop('logged')

    #shows the waterfall plot
    this_plot = plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        **kwargs
    )
    #add colorbar
    if cb:
        plt.colorbar()

    #add plot labels
    if freq_label:
        plt.xlabel("Frequency [Hz]",fontdict=font)
    if MJD_time:
        plt.ylabel("Time [MJD]",fontdict=font)
    else:
        plt.ylabel("Time [s]",fontdict=font)

    return this_plot 

def plot_candidate_events_individually(full_candidate_event_dataframe, correct_fils, source_name, node_string, filter_level, show=False, overwrite=False, offset=0, **kwargs):
    
    #get only the events in the dataframe that are from the right target
    candidate_event_dataframe = full_candidate_event_dataframe.loc[full_candidate_event_dataframe['Source'] == source_name]

    #load in the data for each individual hit
    for i in range(0, len(candidate_event_dataframe)):
        candidate = candidate_event_dataframe.iloc[i]
        source_id = candidate['Source']
        f_mid = candidate['Freq']
        drate = candidate['DriftRate']
                
        #load in the list of .fil files
        filelist = correct_fils    
            
        #calculate the length of the total ABABAB from the fil files' headers
        fil0 = bl.Waterfall(filelist[0], load_data=False)
        t0 = fil0.header['tstart']
        fil6 = bl.Waterfall(filelist[-1], load_data=False)
        t6 = fil6.header['tstart']
        t_elapsed = Time(t6, format='mjd').unix - Time(t0, format='mjd').unix + (fil6.n_ints_in_file -1) * fil6.header['tsamp']

        #calculate the width of the plot based on making sure the full drift is visible
        bw = 2.4*abs(drate)/1e6 * t_elapsed
        bw = np.max((bw, 500./1e6))

        #Get start and stop frequencies based on midpoint and bandwidth
        f_start, f_stop = np.sort((f_mid-bw/2,  f_mid+bw/2))

        #Print useful values
        print('*************************************************')
        print('***     The Parameters for This Plot Are:     ***')
        print('*************************************************')   
        print('Target = ', source_id)
        print('Bandwidth (MHz) = ', bw)
        print('Total Time Elapsed (s) = ', t_elapsed)
        print('Start Frequency (MHz) = ', f_start)
        print('Middle Frequency (MHz) = ', f_mid)
        print('Stop Frequency (MHz) = ', f_stop)
        print('Expected Drift (Hz/s) = ', drate)
        print('*************************************************')
        print('*************************************************')
                
        #Pass info to make_waterfall_plots() function
        subplots = make_waterfall_plots(filelist, 
                                        [source_id], 
                                        [drate], 
                                        [f_mid], 
                                        f_start, 
                                        f_stop, 
                                        node_string,
                                        filter_level,
                                        ion=False, 
                                        epoch=None,
                                        local_host='',
                                        plot_name='',
                                        save_pdf_plot=False,
                                        saving_fig=False, 
                                        offset=offset, 
                                        **kwargs)
    return 