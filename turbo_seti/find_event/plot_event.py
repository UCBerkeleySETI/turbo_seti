#!/usr/bin/env python3

r'''
Backend script to plot drifting, narrowband events in a generalized cadence of
ON-OFF radio SETI observations. The main function contained in this file is
:func:`~.plot_candidate_events` uses the other helper functions
in this file (described below) to plot events from a turboSETI event .csv file.
'''

from os import mkdir
from os.path import dirname
import gc
import logging
logger_plot_event_name = 'plot_event'
logger_plot_event = logging.getLogger(logger_plot_event_name)
logger_plot_event.setLevel(logging.INFO)

# Plotting packages import
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

# Math/Science package imports
import numpy as np
from astropy.time import Time
import pandas as pd

# BL imports
import blimpy as bl
from blimpy.utils import rebin
from . import find_event

# preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)

def overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration, offset=0, alpha=1, c='#cc0000'):
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
                 c=c,
                 lw=2,
                 alpha=alpha)

    # plots drift overlay line, with offset if desired
    plt.plot((f_event + offset, f_event + drift_rate/1e6 * t_duration + offset),
             (0, t_duration),
             c=c,
             ls='dashed', lw=2, 
             alpha=alpha)

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

    # Rolled back PR #82

    # determine extent of the plotting panel for imshow
    extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    # plot and scale intensity (log vs. linear)
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    plot_data = 10.0 * np.log10(plot_data)

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
        Middle frequency of the event, in MHz.
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
    global logger_plot_event

    # prepare for plotting
    matplotlib.rc('font', **font)

    # set up the sub-plots
    n_plots = len(fil_file_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    # get directory path for storing PNG files
    dirpath = dirname(fil_file_list[0]) + '/'

    # read in data for the first panel
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
        logger_plot_event.debug('make_waterfall_plots: file {} in list: {}'.format(i, filename))
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
    path_png = dirpath + filter_level + '_' + on_source_name + '_dr_' + "{:0.2f}".format(drift_rate) + '_freq_' "{:0.6f}".format(f_start) + ".png"
    plt.savefig(path_png, bbox_inches='tight')
    logger_plot_event.debug('make_waterfall_plots: Saved file {}'.format(path_png))

    # close all figure windows
    plt.close('all')

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
    plot_snr_list : bool (*** NOT YET IN USE***)
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
    global logger_plot_event
    
    # load in the data for each individual hit
    if candidate_event_dataframe is None:
        print('*** plot_candidate_events: candidate_event_dataframe is None, nothing to do.')
        return
    len_df = len(candidate_event_dataframe)
    if len_df < 1:
        print('*** plot_candidate_events: len(candidate_event_dataframe) = 0, nothing to do.')
        return
    for i in range(0, len_df):
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

        # logger_plot_event.debug useful values
        logger_plot_event.debug('*************************************************')
        logger_plot_event.debug('***     The Parameters for This Plot Are:    ****')
        logger_plot_event.debug('Target = {}'.format(on_source_name))
        logger_plot_event.debug('Bandwidth = {} MHz'.format(round(bandwidth, 5)))
        logger_plot_event.debug('Time Elapsed (inc. Slew) = {} s'.format(round(t_elapsed)))
        logger_plot_event.debug('Middle Frequency = {} MHz'.format(round(f_mid, 4)))
        logger_plot_event.debug('Expected Drift = {} Hz/s'.format(round(drift_rate, 4)))
        logger_plot_event.debug('*************************************************')

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

def plot_all_hit_and_candidates(dat_list_string, fils_list_string, candidate_event_table_string, 
                                outdir=None, check_zero_drift=False, alpha=1, c='#cc0000', window=None):
    """
    Makes a plot similar to the one produced by 
    plot_candidate_events, but also includes the hits 
    detected, in addition to the candidate signal.
    
    Calls `plot_all_dat` and `plot_dat`
    
    Arguments
    ----------
    dat_list_string : str
        List of .dat files in the cadence.
    fils_list_string : str
        List of filterbank or .h5 files in the cadence.
    candidate_event_table_string : str
        The string name of a .csv file that contains the
        list of events at a given filter level, created as
        output from find_event_pipeline.py. 
    outdir : str, optional
        Path to the directory where the plots will be saved to. 
        The default is None, which will result in the plots being 
        saved to the directory where the .dat file are located.
    check_zero_drift : bool, optional
         A True/False flag that tells the program whether to
        include hits that have a drift rate of 0 Hz/s. Earth-
        based RFI tends to have no drift rate, while signals
        from the sky are expected to have non-zero drift rates.
        The default is False.
    outdir : str, optional
        Path to the directory where the plots will be saved to. 
        The default is None, which will result in the plots being 
        saved to the directory the .dat file are located.
    alpha : float, optional
        The opacity of the overlayed hit plot. This should 
        be between 0 and 1, with 0 being invisible, and 1
        being the default opacity. This is passed into 
        matplotlib.pyplot function. 
    c : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default color is identical to the color of the 
        candidate signal plot, so choosing a different color is 
        recommended. The default is '#cc0000'.
    window : tuple, optional
        Sets the start and stop frequencies of the plot, in MHz.
        The input takes the form of a tuple: (start, stop). And 
        assumes that the start is less than the stop. If given, the 
        resulting plot will range exactly between the start/stop
        frequencies. The default is None, which will result in 
        a plot of the entire range of hits detected.
    """
    #read candidate events into dataframe
    candidate_event_dataframe = pd.read_csv(candidate_event_table_string)
    
    #read in dat files
    dat_file_list = []
    for file in pd.read_csv(dat_list_string, encoding='utf-8', header=None, chunksize=1):
        dat_file_list.append(file.iloc[0,0])
        
    # put all hits into a single dataframe
    all_hits = []
    for dat in dat_file_list:
        frame = find_event.read_dat(dat)
        all_hits.append(frame)
    all_hits_frame = pd.concat(all_hits)
    
    #change the min/max frequency if specified
    if window != None:
        f_min = window[0]
        f_max = window[1]
        keep = np.where((all_hits_frame["Freq"] >= f_min) & (all_hits_frame["Freq"] <= f_max))
        all_hits_frame = all_hits_frame.iloc[keep]
        
        #keep only the candidates within the window 
        keep = np.where((candidate_event_dataframe["Freq"] > f_min) & (candidate_event_dataframe["Freq"] < f_max))
        candidate_event_dataframe = candidate_event_dataframe.iloc[keep]
        
    #read in fil files
    fil_file_list = []
    for file in pd.read_csv(fils_list_string, encoding='utf-8', header=None, chunksize=1):
        fil_file_list.append(file.iloc[0,0])
        
    #obtaining source names
    source_name_list = []
    for fil in fil_file_list:
        wf = bl.Waterfall(fil, load_data=False)
        source_name = wf.container.header["source_name"]
        source_name_list.append(source_name)
        print("plot_all_dat: source_name={}".format(source_name))
        
    n_events = len(candidate_event_dataframe)
    if n_events == 0:
        # check to see if there are any hits detected
        if len(all_hits_frame) == 0:
            print("There are no hits in this range. This will make 0 .png files")
        else:
            # plot just the hits as there will be no candidate
            print("This will make 1 .png file")
            plot_all_dat(dat_file_list, 
                         fil_file_list, 
                         source_name_list,
                         all_hits_frame,
                         outdir=outdir,
                         check_zero_drift=check_zero_drift, 
                         alpha=alpha, 
                         c=c,
                         window=window)
    else:
        #plot the hits and candidate(s)
        print("This will make %s .png files"%n_events)
        for i in range(n_events):
            candidate = candidate_event_dataframe.iloc[i]
            plot_all_dat(dat_file_list, 
                         fil_file_list, 
                         source_name_list,
                         all_hits_frame,
                         candidate,
                         outdir=outdir,
                         check_zero_drift=check_zero_drift, 
                         alpha=alpha, 
                         c=c,
                         window=window)
        
def plot_all_dat(dat_file_list, fil_file_list, source_name_list, all_hits_frame, candidate=None, check_zero_drift=False, 
                 outdir=None, alpha=1, c='#cc0000', window=None):
    """ 
    Parameters
    ----------
    dat_file_list : list
        A Python list that contains a series of 
        strings corresponding to the filenames of .dat
        files, each on a new line, that corresponds to 
        the .dat files created when running turboSETI 
        candidate search on the .h5 or .fil files below
    fil_file_list : list
        A Python list that contains a series of 
        strings corresponding to the filenames of .dat
        files, each on a new line, that corresponds to 
        the cadence used to create the .csv file used 
        for event_csv_string.
    source_name_list : list
        A Python list that contains a series of strings
        corresponding to the source names of the 
        cadence in chronological (descending through 
        the plot pannels) cadence.
    all_hits_frame : dict
        A pandas dataframe contining information about 
        all the hits detected. The necessary data 
        includes the start and stop frequencies, the drift
        rate, and the source name. This dataframe is 
        generated in plot_all_hit_and_candidates above. 
    candidate : dict, optional
        A single row from a pandas dataframe containing
        information about one of the candidate signals
        detected. Contains information about the candidate 
        signal to be plotted. The necessary data includes 
        the start and stop frequencies, the drift rate, 
        and the source name. The dataframe the candiate
        comes from is generated in plot_all_hit_and_candidates
        above as `candidate_event_dataframe`. The default is None.
    check_zero_drift : bool, optional
         A True/False flag that tells the program whether to
        include hits that have a drift rate of 0 Hz/s. Earth-
        based RFI tends to have no drift rate, while signals
        from the sky are expected to have non-zero drift rates.
        The default is False.
    outdir : str, optional
        Path to the directory where the plots will be saved to. 
        The default is None, which will result in the plots being 
        saved to the directory the .dat file are located.
    alpha : float, optional
        The opacity of the overlayed hit plot. This should 
        be between 0 and 1, with 0 being invisible, and 1
        being the default opacity. This is passed into 
        matplotlib.pyplot function. 
    c : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default color is identical to the color of the 
        candidate signal plot, so choosing a different color is 
        recommended. The default is '#cc0000'.
    window : tuple, optional
        Sets the start and stop frequencies of the plot, in MHz.
        The input takes the form of a tuple: (start, stop). And 
        assumes that the start is less than the stop. The 
        resulting plot will range exactly between the start/stop
        frequencies. The default is None, which will result in 
        a plot of the entire range of hits detected.
    """
    #set plot boundaries based on the contents of the file
    freq_range = np.max(all_hits_frame["Freq"]) - np.min(all_hits_frame["Freq"])
    filter_level = "f0"
    
    # total range all hits fall between 
    f_min = np.min(all_hits_frame["Freq"])
    f_max = np.max(all_hits_frame["Freq"])
    
    
    fil1 = bl.Waterfall(fil_file_list[0], load_data=False)
    t0 = fil1.header["tstart"]
    t_elapsed = Time(fil1.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
    bandwidth = 2.4 * abs(freq_range)/1e6 * t_elapsed
    bandwidth = np.max((bandwidth, 500./1e6))
    
    # Get start and stop frequencies based on midpoint and bandwidth
    f_start, f_stop = np.sort((f_min - (bandwidth/2),  f_max + (bandwidth/2)))
    mid_f = 0.5*(f_start + f_stop)
    
    #if given a window to plot in, set boundaries accordingly
    if window is not None:
        f_start = window[0]
        f_stop  = window[1]
        mid_f   = 0.5*(f_start + f_stop) 
    
    # plugging some code from make_waterfall_plots
    global logger_plot_event

    # prepare for plotting
    matplotlib.rc('font', **font)

    # set up the sub-plots
    n_plots = len(fil_file_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    # get directory path for storing PNG files
    dirpath = dirname(fil_file_list[0]) + '/'

    # read in data for the first panel
    max_load = bl.calcload.calc_max_load(fil_file_list[0])
    print('plot_event make_waterfall_plots: max_load={} is required for {}'.format(max_load, fil_file_list[0]))
    fil1 = bl.Waterfall(fil_file_list[0], f_start=f_start, f_stop=f_stop, max_load=max_load)
    t0 = fil1.header['tstart']
    dummy, plot_data1 = fil1.grab_data()

    # rebin data to plot correctly with fewer points
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data1.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  int(np.ceil(plot_data1.shape[1] /  MAX_IMSHOW_POINTS[1]))
    plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)
    
    subplots = []
    del fil1, dummy, plot_data1
    gc.collect()
    
    on_source_name = source_name_list[0]
    f_candidate = mid_f
    
    if candidate is not None:
        on_source_name = candidate["Source"]
        f_candidate = candidate["Freq"]
    
    for i, filename in enumerate(fil_file_list):
        subplot = plt.subplot(n_plots, 1, i+1)
        subplots.append(subplot)
        
        #read in the data
        max_load = bl.calcload.calc_max_load(filename)
        print('plot_event make_waterfall_plots: max_load={} is required for {}'.format(max_load, filename))
        wf = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop, max_load=max_load)
        
        this_plot = plot_waterfall(wf, 
                                   source_name_list[i], 
                                   f_start, 
                                   f_stop)
        
        plot_dat(dat_file_list[i], fil_file_list[i],
                 f_start, f_stop, t0, candidate, check_zero_drift=check_zero_drift, alpha=alpha, c=c)
        
        #more code from make_waterfall_plots
        # Title the full plot
        if i == 0:
            plot_title = "%s \n MJD:%5.5f" % (on_source_name, t0)

            plt.title(plot_title)
        # Format full plot
        if i < len(fil_file_list)-1:
            plt.xticks(np.linspace(f_start, f_stop, num=4), ['','','',''])

        del wf
        gc.collect()        

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
    if outdir is not None:
        dirpath = outdir
        
    # make note if the plot contains a candidate
    cand = ""
    if candidate is not None:
        cand = "_candidate"
    path_png = dirpath + filter_level + '_' + on_source_name + cand + '_freq_' "{:0.6f}".format(f_candidate) + ".png"
    plt.savefig(path_png, bbox_inches='tight', transparent=False)
    logger_plot_event.debug('make_waterfall_plots: Saved file {}'.format(path_png))

    # close all figure windows
    plt.close('all')

def plot_dat(dat, fil, f_start, f_stop, t0, candidate=None, check_zero_drift=False, alpha=1, c='#cc0000'):
    """
    Parameters
    ----------
    dat : str
        The .dat file containing information about the hits detected.
    fil : str
        Filterbank or h5 file corresponding to the .dat file.
    f_start : float
        Start frequency, in MHz.
    f_stop : float
        Stop frequency, in MHz.
    t0 : float
        Start time of the candate event in mjd units.
    candidate : dict, optional
        A single row from a pandas dataframe containing
        information about one of the candidate signals
        detected. Contains information about the candidate 
        signal to be plotted. The necessary data includes 
        the start and stop frequencies, the drift rate, 
        and the source name. The dataframe the candiate
        comes from is generated in plot_all_hit_and_candidates
        above as `candidate_event_dataframe`. The default is None.
    check_zero_drift : bool, optional
         A True/False flag that tells the program whether to
        include hits that have a drift rate of 0 Hz/s. Earth-
        based RFI tends to have no drift rate, while signals
        from the sky are expected to have non-zero drift rates.
        The default is False.
    alpha : float, optional
        The opacity of the overlayed hit plot. This should 
        be between 0 and 1, with 0 being invisible, and 1
        being the default opacity. This is passed into 
        matplotlib.pyplot function. 
    c : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default color is identical to the color of the 
        candidate signal plot, so choosing a different color is 
        recommended. The default is '#cc0000'.

    """
    wf = bl.Waterfall(fil, f_start, f_stop)
    hit_frame = find_event.read_dat(dat)
    
    #select just the hits within the frequency range
    if len(hit_frame) > 0:
        keep = np.where((hit_frame['Freq'] > f_start) & (hit_frame['Freq'] < f_stop))
        hit_frame = hit_frame.iloc[keep]
    
    # plot the estimated candidate line 
    if candidate is not None:
        t_elapsed = Time(wf.header['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
        t_duration = (wf.n_ints_in_file - 1) * wf.header['tsamp']
        f_event = candidate["Freq"] + candidate["DriftRate"] / 1e6 * t_elapsed
        drift_rate = candidate["DriftRate"]
        # any mistakes will likely come from this line
        overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration)
    
    if len(hit_frame) == 0:
        # there are no hits detected in this dat file 
        return
    
    if not check_zero_drift:
        hit_frame = hit_frame[hit_frame["DriftRate"] != 0]
    
    #f_mid = 0.5 * (f_start + f_stop)
    t_duration = (wf.n_ints_in_file - 1) * wf.header["tsamp"]
    
    #plot all the hits 
    for i in range(len(hit_frame)):
        hit = hit_frame.iloc[i]
        
        f_mid = hit["Freq"]
        drift_rate = hit["DriftRate"]
        f_event = f_mid 
        bandwidth = 500./1e6
        start, stop = np.sort((f_mid - (bandwidth/2),  f_mid + (bandwidth/2)))
        
        overlay_drift(f_event, start, stop, drift_rate, t_duration, offset=0, alpha=alpha, c=c)