#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part of the Breakthrough Listen software package turboSETI

Backend script to plot drifting, narrowband events in a generalized cadence of 
ON-OFF radio SETI observations.

The main function contained in this file is *plot_candidate_events*
    Plot_candidate_events uses the other helper functions in this file 
    (described below) to plot events from a turboSETI event .csv file. 
    
The following helper functions are contained in this file:
    overlay_drift           - creates a dashed red line at the recorded 
                              frequency and drift rate of the plotted event -
                              can overlay the signal exactly or be offset by 
                              some amount (offset can be 0 or 'auto')
    plot_waterfall          - makes a single-panel waterfall plot (frequency 
                              vs. time vs. intensity) for one of the on or off 
                              observations in the cadence of interest, at the 
                              frequency of the expected event. Calls 
                              overlay_drift
    make_waterfall_plots    - makes a series of waterfall plots, to be read 
                              from top to bottom, displaying a full cadence 
                              at the frequency of a recorded event from 
                              find_event. Calls plot_waterfall
    plot_candidate_events   - Calls make_waterfall_plots on each event in the 
                              input .csv file
                      
Usage (beta):
    It is highly recommended that users interact with this program via the
    front-facing plot_event_pipeline.py script. See the usage of that file in
    its own documentation. 
    
    If you would like to run plot_candidate_events without calling
    plot_event_pipeline.py, the usage is as follows:
    
    plot_event.plot_candidate_events(candidate_event_dataframe, 
                                     fil_file_list, 
                                     filter_level, 
                                     source_name_list,
                                     offset=0)
        
    candidate_event_dataframe   A pandas dataframe containing information
                                about a candidate event. The necessary data
                                includes the start and stop frequencies, the
                                drift rate, and the source name. To determine
                                the required variable names and formatting
                                conventions, see the output of
                                find_event_pipeline.
                   
    fil_file_list               A Python list that contains a series of 
                                strings corresponding to the filenames of .fil 
                                files, each on a new line, that corresponds to 
                                the cadence used to create the .csv file used 
                                for event_csv_string.
                   
    filter_level                A string indicating the filter level of the 
                                cadence used to generate the 
                                candidate_event_dataframe. Used only for
                                output file naming, convention is "f1", "f2",
                                or "f3". Descriptions for the three levels of
                                filtering can be found in the documentation
                                for find_event.py

    source_name_list            A Python list that contains a series of strings
                                corresponding to the source names of the
                                cadence in chronological (descending through
                                the plot panels) cadence.
                        
    offset                      The amount that the overdrawn "best guess" 
                                line from the event parameters in the csv 
                                should be shifted from its original position 
                                to enhance readability. Can be set to 0 
                                (default; draws line on top of estimated
                                event) or 'auto' (shifts line to the left by 
                                an auto-calculated amount, with addition lines 
                                showing original position).

author: 
    Version 2.0 - Sofia Sheikh (ssheikhmsa@gmail.com)
    Version 1.0 - Emilio Enriquez (jeenriquez@gmail.com)
    
Last updated: 05/24/2020

"""

#General packages import
import glob
import os
import sys
import pandas as pd
#import pylab as plt
import numpy as np
import logging; logging.disable(logging.CRITICAL);

#BL + my packages import
#import updated_find_event
import pandas as pd
import blimpy as bl
from blimpy.utils import rebin
from .find_event import make_table

#Plotting packages import
import matplotlib
import matplotlib.pyplot as plt

#preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)


def plot_hit(fil_filename, dat_filename, hit_id, bw=None, offset=0):
    """Plot a candidate from a .dat file

    Args:
      fil_filename(str): Path to filterbank file to plot
      dat_filename(str): Path to turbosSETI generated .dat output file of events
      hit_id(int): ID of hit in the dat file to plot (TopHitNum)
      offset(float, optional): Offset drift line on plot. Default 0.
      bw:  (Default value = None)

    Returns:

    """
    # Load hit details
    dat = make_table(dat_filename)
    hit = dat.iloc[hit_id]

    f0 = hit['Freq']

    if bw is None:
        bw_mhz = np.abs(hit['FreqStart'] - hit['FreqEnd'])
    else:
        bw_mhz = bw * 1e-6

    fil = bl.Waterfall(fil_filename, f_start=f0 - bw_mhz / 2, f_stop=f0 + bw_mhz / 2)
    t_duration = (fil.n_ints_in_file - 1) * fil.header['tsamp']

    fil.plot_waterfall()
    overlay_drift(f0, hit['DriftRate'], t_duration, offset)


def rebin(d, n_x, n_y):
    """Rebin data by averaging bins together

    Args:
      d(np.array): data
      n_x(int): number of bins in x dir to rebin into one
      n_y: 

    Returns:
      d: rebinned data with shape (n_x, n_y)

    """

    if d.ndim == 2:
        if n_y is None:
            n_y = 1
        if n_x is None:
            n_x = 1
        print ('d.shape[0]', d.shape[0])
        print ('d.shape[1]', d.shape[1])
        print ('n_x', n_x)
        print ('n_y', n_y)
        d = d[:int(d.shape[0] // n_x) * n_x, :int(d.shape[1] // n_y) * n_y]
        d = d.reshape((d.shape[0] // n_x, n_x, d.shape[1] // n_y, n_y))
        d = d.mean(axis=3)
        d = d.mean(axis=1)
    elif d.ndim == 1:
        d = d[:int(d.shape[0] // n_x) * n_x]
        d = d.reshape((d.shape[0] // n_x, n_x))
        d = d.mean(axis=1)
    else:
        raise RuntimeError("Only NDIM <= 2 supported")
    return d


def overlay_drift(f_event, drate, t_duration, offset=0):
    """Overlay drift line on plot

    Args:
      f_event(float): Start frequency of event in MHz
      drate(float): Drift rate in Hz/s
      t_duration(float): Time duration in seconds
      offset(float, optional): Offset line from event by amount. Default 0.

    Returns:

    """
    if offset == 'auto':
        offset = - 0.2 * drate*t_duration
    plt.plot((f_event+offset/1e6, f_event+drate/1e6*t_duration+offset/1e6), (0, t_duration), c='#cc0000', ls='dashed', lw=2)


def make_waterfall_plots(filenames_list, target, drates, fvals, f_start,f_stop, node_string, filter_level, ion=False,
                         epoch=None,bw=250.0, local_host='',plot_name='',save_pdf_plot=False,saving_fig=False,offset=0,
                         dedoppler=False,**kwargs):
    """Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)

    Args:
      filenames_list: 
      target: 
      drates: 
      fvals: 
      f_start: 
      f_stop: 
      node_string: 
      filter_level: 
      ion:  (Default value = False)
      epoch:  (Default value = None)
      bw:  (Default value = 250.0)
      local_host:  (Default value = '')
      plot_name:  (Default value = '')
      save_pdf_plot:  (Default value = False)
      saving_fig:  (Default value = False)
      offset:  (Default value = 0)
      dedoppler:  (Default value = False)
      **kwargs: 

    Returns:

    """
    
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


    #finding plotting values range for the first panel (A1)
    fil = bl.Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)

    t0 = fil.header['tstart']
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)
    dec_fac_x, dec_fac_y = 1, 1
    
    #rebinning data to plot correctly with fewer points
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] // MAX_IMSHOW_POINTS[0] 
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
        print ('dec_fac_y', dec_fac_y)
    #insert rebin definition code from utils.py in blimpy and edit it because this is the problem
    d=plot_data
    n_x=dec_fac_x
    n_y=dec_fac_y

    d = rebin(d, n_x, n_y)
    plot_data=d
    #print('d', d)

    #plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

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
        
        #read in data
        fil = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop)


        try:
            #make plot with plot_waterfall
            source_name = source_name_list[i]
            this_plot = plot_waterfall(fil,
                                       source_name,
                                       f_start=f_start, 
                                       f_stop=f_stop, 
                                       drift_rate=drift_rate,
                                       **kwargs)
            
            #calculate parameters for estimated drift line
            t_elapsed = Time(fil.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
            t_duration = (fil.n_ints_in_file - 1) * fil.header['tsamp']
            f_event = f_mid + drift_rate / 1e6 * t_elapsed
            
            #plot estimated drift line
            overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration, offset)
        except:
            raise
            
        #Title the full plot
        if i == 0:
            plot_title = "%s \n $\dot{\\nu}$ = %2.3f Hz/s" % (on_source_name, drift_rate)
            plt.title(plot_title)
        #Format full plot
        if i < len(fil_file_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #More overall plot formatting, axis labelling
    factor = 1e6
    units = 'Hz'
            
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    xloc = np.linspace(f_start, f_stop, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
    if np.max(xticks) > 1000:
        xticks = [xt/1000 for xt in xticks]
        units = 'kHz'
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)
    
    #Add colorbar
    cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Normalized Power')
    
    #Adjust plots
    plt.subplots_adjust(hspace=0,wspace=0)
    
    #save the figures
    plt.savefig(filter_level + '_' + on_source_name + '_dr_' + "{:0.2f}".format(drift_rate) + '_freq_' "{:0.6f}".format(f_start) + ".png",
               bbox_inches='tight')

    return subplots


def plot_waterfall(fil, f_start=None, f_stop=None, drate=None, if_id=0, logged=True, cb=False,freq_label=False,MJD_time=False, **kwargs):
    """Plot waterfall of data

    Args:
      f_start(float, optional): start frequency, in MHz (Default value = None)
      f_stop(float, optional): stop frequency, in MHz (Default value = None)
      logged(bool, optional): Plot in linear (False) or dB units (True), (Default value = True)
      cb(bool, optional): for plotting the colorbar (Default value = False)
      kwargs: keyword args to be passed to matplotlib imshow()
      fil: 
      drate:  (Default value = None)
      if_id:  (Default value = 0)
      freq_label:  (Default value = False)
      MJD_time:  (Default value = False)
      **kwargs: 

    Returns:

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
    #insert rebin definition code from utils.py in blimpy and edit it because this is the problem
    d=plot_data
    n_x=dec_fac_x
    n_y=dec_fac_y

    d = rebin(d, n_x, n_y)
    plot_data=d
    #print('d', d)

    #plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

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

def plot_candidate_events_individually(full_candidate_event_dataframe, correct_fils, source_name,
                                       node_string, filter_level, show=False, overwrite=False, offset=0, **kwargs):
    """

    Args:
      full_candidate_event_dataframe: 
      correct_fils: 
      source_name: 
      node_string: 
      filter_level: 
      show:  (Default value = False)
      overwrite:  (Default value = False)
      offset:  (Default value = 0)
      **kwargs: 

    Returns:

    """
    trimmed_candidate_event_dataframe=pd.read_csv(full_candidate_event_dataframe)
    print ('full_candidate_event_dataframe', full_candidate_event_dataframe)
    #get only the events in the dataframe that are from the right target
    #full_candidate_event_dataframe=full_candidate_event_dataframe['Source']
    print ('trimmed_candidate_event_dataframe_Source', trimmed_candidate_event_dataframe.loc[:, ['Source']])
    #print ('full_candidate_event_dataframe_Source', full_candidate_event_dataframe['Source'])
    candidate_event_dataframe = trimmed_candidate_event_dataframe.loc[:, ['Source', 'Freq', 'DriftRate','FreqStart']]
    #candidate_event_dataframe = full_candidate_event_dataframe.loc[full_candidate_event_dataframe['Source'] == source_name]
    print ('full_candidate_event_dataframe', candidate_event_dataframe)

    #load in the data for each individual hit
    for i in range(0, len(candidate_event_dataframe)):
        candidate = candidate_event_dataframe.iloc[i]
        source_id = candidate['Source']
        f_mid = candidate['Freq']
        drate = -1*candidate['DriftRate']
                
        #load in the list of .fil files
        filelist = open(correct_fils).readlines()
        filelist = [files.replace('\n','') for files in filelist]
        #print ('dat_file_list', dat_file_list)
        filelist = [files.replace(',','') for files in filelist]
        #filelist = correct_fils    
        print ('filelist', filelist) 
        #calculate the length of the total ABABAB from the fil files' headers
        fil0 = bl.Waterfall(filelist[0], load_data=True)
        print ('fil0', fil0)
        fil0header=fil0.header
        print ('fil0header', fil0header)
        y={}
        for key, value in fil0header.items():
            y[key.decode("utf-8")]=value
        fil0header=y
        print ('fil0header', fil0header)
        t0 = fil0header['tstart']
        
        fil6 = bl.Waterfall(filelist[-1], load_data=True)
        fil6header=fil6.header
        z={}
        for key, value in fil6header.items():
            z[key.decode("utf-8")]=value
        fil6header=z
        t6 = fil6header['tstart']
        t_elapsed = Time(t6, format='mjd').unix - Time(t0, format='mjd').unix + (fil6.n_ints_in_file -1) * fil6header['tsamp']


        #calculate the width of the plot based on making sure the full drift is visible
        bw = 2.4*abs(drate)/1e6 * t_elapsed
        bw = np.max((bw, 500./1e6))

        #Get start and stop frequencies based on midpoint and bandwidth
        f_start, f_stop = np.sort((f_mid-bw/2,  f_mid+bw/2))

        #Print useful values
        print('')
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
        print('')
        
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
                                        save_pdf_plot=True,
                                        saving_fig=True, 
                                        offset=offset, 
                                        **kwargs)
    return 
