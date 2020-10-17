#!/usr/bin/env python3

r'''
Backend script to plot drifting, narrowband events in a generalized cadence of
ON-OFF radio SETI observations. The main function contained in this file is
:func:`~.plot_candidate_events` uses the other helper functions
in this file (described below) to plot events from a turboSETI event .csv file.
'''

from os.path import dirname

import matplotlib
matplotlib.use('agg')

# General packages import
import numpy as np
import logging; logging.disable(logging.CRITICAL)
from astropy.time import Time

# BL imports
import blimpy as bl
from blimpy.utils import rebin

# Plotting packages import
import matplotlib.pyplot as plt

# preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)

def overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration, offset=0):
    r'''
    Creates a dashed red line at the recorded frequency and drift rate of
    the plotted event - can overlay the signal exactly or be offset by
    some amount (offset can be 0 or 'auto').

    '''
    # determines automatic offset and plots offset lines
    if offset == 'auto':
        offset = ((f_start - f_stop) / 10)
        plt.plot((f_event - offset, f_event),
                 (10, 10),
                 "o-",
                 c='#cc0000',
                 lw=2)

    # plots drift overlay line, with offset if desired
    plt.plot((f_event + offset, f_event + drift_rate/1e6 * t_duration + offset),
             (0, t_duration),
             c='#cc0000',
             ls='dashed', lw=2)

def plot_waterfall(fil, source_name, f_start=None, f_stop=None, **kwargs):
    r"""
    Plot waterfall of data in a .fil or .h5 file.
    
    Parameters
    ----------
    fil : str
        Filterbank file containing the dynamic spectrum data.
    source_name : str
        Name of the target.
    f_start : float
        Start frequency, in MHz.
    f_stop : float
        Stop frequency, in MHz.
    kwargs : dict
        Keyword args to be passed to matplotlib imshow().

    Notes
    -----
    Plot a single-panel waterfall plot (frequency vs. time vs. intensity)
    for one of the on or off observations in the cadence of interest, at the
    frequency of the expected event. Calls :func:`~overlay_drift`

    """
    # prepare font
    matplotlib.rc('font', **font)

    # Load in the data from fil
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1

    # rebinning data to plot correctly with fewer points
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  int(np.ceil(plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]))
    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    # fix case where frequencies are reversed by fil.grab_data() # Shane Smith PR #82
    if plot_f[-1] < plot_f[0]:
        plot_f = plot_f[::-1]

    # determine extent of the plotting panel for imshow
    extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    # plot and scale intensity (log vs. linear)
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    kwargs['logged'] = True
    if kwargs['logged'] == True:
        plot_data = 10*np.log10(plot_data)
        kwargs.pop('logged')

    # get normalization parameters
    vmin = plot_data.min()
    vmax = plot_data.max()
    normalized_plot_data = (plot_data - vmin) / (vmax - vmin)

    # display the waterfall plot
    this_plot = plt.imshow(normalized_plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        **kwargs
    )

    # add plot labels
    plt.xlabel("Frequency [Hz]",fontdict=font)
    plt.ylabel("Time [s]",fontdict=font)

    # add source name
    ax = plt.gca()
    plt.text(0.03, 0.8, source_name, transform=ax.transAxes, bbox=dict(facecolor='white'))
    # if plot_snr != False:
    #     plt.text(0.03, 0.6, plot_snr, transform=ax.transAxes, bbox=dict(facecolor='white'))
    # return plot
    return this_plot

def make_waterfall_plots(fil_file_list, on_source_name, f_start, f_stop, drift_rate, f_mid,
                         filter_level, source_name_list, offset=0, **kwargs):
    r'''
    Makes waterfall plots of an event for an entire on-off cadence.
    
    Parameters
    ----------
    fil_file_list : str
        List of filterbank files in the cadence.
    on_source_name : str
        Name of the on_source target.
    f_start : float
        Start frequency, in MHz.
    f_stop : float
        Stop frequency, in MHz.
    drift_rate : float
        Drift rate in Hz/s.
    f_mid : float
        <iddle frequency of the event, in MHz.
    filter_level : int
        Filter level (1, 2, or 3) that produced the event.
    source_name_list : list
        List of source names in the cadence, in order.
    bandwidth : int
        Width of the plot, incorporating drift info.
    kwargs : dict
        Keyword args to be passed to matplotlib imshow().

    Notes
    -----
    Makes a series of waterfall plots, to be read from top to bottom, displaying a full cadence
    at the frequency of a recorded event from find_event. Calls :func:`~plot_waterfall`

    '''
    # prepare for plotting
    matplotlib.rc('font', **font)

    # set up the sub-plots
    n_plots = len(fil_file_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))
    
    # get directory path for storing PNG files
    dirpath = dirname(fil_file_list[0]) + '/'

    # read in data for the first panel
    print('make_waterfall_plots first_file in list: {}'.format(fil_file_list[0]))
    fil1 = bl.Waterfall(fil_file_list[0], f_start=f_start, f_stop=f_stop)
    t0 = fil1.header['tstart']
    dummy, plot_data1 = fil1.grab_data()

    # rebin data to plot correctly with fewer points
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data1.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  int(np.ceil(plot_data1.shape[1] /  MAX_IMSHOW_POINTS[1]))
    plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)

    # define more plot parameters
    # never used: delta_f = 0.000250
    mid_f = np.abs(f_start+f_stop)/2.

    subplots = []

    # Fill in each subplot for the full plot
    for i,filename in enumerate(fil_file_list):
        print('make_waterfall_plots file {} in list: {}'.format(i, filename))
        # identify panel
        subplot = plt.subplot(n_plots,1,i+1)
        subplots.append(subplot)

        # read in data
        fil = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop)
        # make plot with plot_waterfall
        source_name = source_name_list[i]
        this_plot = plot_waterfall(fil,
                                   source_name,
                                   f_start=f_start,
                                   f_stop=f_stop,
                                   **kwargs)

        # calculate parameters for estimated drift line
        t_elapsed = Time(fil.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
        t_duration = (fil.n_ints_in_file - 1) * fil.header['tsamp']
        f_event = f_mid + drift_rate / 1e6 * t_elapsed

        # plot estimated drift line
        overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration, offset)

        # Title the full plot
        if i == 0:
            plot_title = "%s \n $\dot{\\nu}$ = %2.3f Hz/s , MJD:%5.5f" % (on_source_name, drift_rate, t0)

            plt.title(plot_title)
        # Format full plot
        if i < len(fil_file_list)-1:
            plt.xticks(np.linspace(f_start, f_stop, num=4), ['','','',''])

    # More overall plot formatting, axis labelling
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

    # Add colorbar
    cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Normalized Power (Arbitrary Units)')

    # Adjust plots
    plt.subplots_adjust(hspace=0,wspace=0)

    # save the figures
    # Somehow, some way, "testing" is inserted between dirpath and the actual file name.  How?????
    plt.savefig(dirpath + filter_level + '_' + on_source_name + '_dr_' + "{:0.2f}".format(drift_rate) + '_freq_' "{:0.6f}".format(f_start) + ".png",
                bbox_inches='tight')

    plt.clf()

    return subplots

def plot_candidate_events(candidate_event_dataframe, fil_file_list, filter_level, source_name_list,
                          offset=0, plot_snr_list=False, **kwargs):
    r'''
    Calls :func:`~make_waterfall_plots` on each event in the input .csv file.

    Arguments
    ---------
    candidate_event_dataframe : dict
        A pandas dataframe containing information
        about a candidate event. The necessary data
        includes the start and stop frequencies, the
        drift rate, and the source name. To determine
        the required variable names and formatting
        conventions, see the output of
        find_event_pipeline.
    fil_file_list : list
        A Python list that contains a series of
        strings corresponding to the filenames of .fil
        files, each on a new line, that corresponds to
        the cadence used to create the .csv file used
        for event_csv_string.
    filter_level : int
        A string indicating the filter level of the
        cadence used to generate the
        candidate_event_dataframe. Used only for
        output file naming, convention is "f1", "f2",
        or "f3". Descriptions for the three levels of
        filtering can be found in the documentation
        for find_event.py
    source_name_list : list
        A Python list that contains a series of strings
        corresponding to the source names of the
        cadence in chronological (descending through
        the plot panels) cadence.
    offset : int, optional
        The amount that the overdrawn "best guess"
        line from the event parameters in the csv
        should be shifted from its original position
        to enhance readability. Can be set to 0
        (default; draws line on top of estimated
        event) or 'auto' (shifts line to the left by
        an auto-calculated amount, with addition lines
        showing original position).
    plot_snr_list : bool
    kwargs : dict


    Examples
    --------
    It is highly recommended that users interact with this program via the
    front-facing plot_event_pipeline.py script. See the usage of that file in
    its own documentation.

    If you would like to run plot_candidate_events without calling
    plot_event_pipeline.py, the usage is as follows:

    >>> plot_event.plot_candidate_events(candidate_event_dataframe, fil_file_list,
    ...                                  filter_level, source_name_list, offset=0)

    '''
    # load in the data for each individual hit
    for i in range(0, len(candidate_event_dataframe)):
        candidate = candidate_event_dataframe.iloc[i]
        on_source_name = candidate['Source']
        f_mid = candidate['Freq']
        drift_rate = candidate['DriftRate']

        # calculate the length of the total cadence from the fil files' headers
        first_fil = bl.Waterfall(fil_file_list[0], load_data=False)
        tfirst = first_fil.header['tstart']
        last_fil = bl.Waterfall(fil_file_list[-1], load_data=False)
        tlast = last_fil.header['tstart']
        t_elapsed = Time(tlast, format='mjd').unix - Time(tfirst, format='mjd').unix + (last_fil.n_ints_in_file -1) * last_fil.header['tsamp']

        # calculate the width of the plot based on making sure the full drift is visible
        bandwidth = 2.4 * abs(drift_rate)/1e6 * t_elapsed
        bandwidth = np.max((bandwidth, 500./1e6))

        # Get start and stop frequencies based on midpoint and bandwidth
        f_start, f_stop = np.sort((f_mid - (bandwidth/2),  f_mid + (bandwidth/2)))

        # Print useful values
        print('')
        print('*************************************************')
        print('***     The Parameters for This Plot Are:    ****')
        print('Target = ', on_source_name)
        print('Bandwidth = ', round(bandwidth, 5), ' MHz')
        print('Time Elapsed (inc. Slew) = ', round(t_elapsed), ' s')
        print('Middle Frequency = ', round(f_mid, 4), " MHz")
        print('Expected Drift = ', round(drift_rate, 4), " Hz/s")
        print('*************************************************')
        print('*************************************************')
        print('')

        # Pass info to make_waterfall_plots() function
        make_waterfall_plots(fil_file_list,
                             on_source_name,
                             f_start,
                             f_stop,
                             drift_rate,
                             f_mid,
                             filter_level,
                             source_name_list,
                             offset=offset,
                             **kwargs)
 
