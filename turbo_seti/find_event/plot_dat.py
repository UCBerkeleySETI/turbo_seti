from os import mkdir
from os.path import dirname
import gc
import logging
logger_plot_event_name = 'plot_dat'
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
from . import plot_event

# preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 1268)

def plot_dat(dat_list_string, fils_list_string, candidate_event_table_string, 
                                outdir=None, check_zero_drift=False, alpha=1, color='black', window=None):
    """
    Makes a plot similar to the one produced by 
    plot_candidate_events, but also includes the hits 
    detected, in addition to the candidate signal.
    
    Calls `plot_hit_candidate` and `make_plot`
    
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
    color : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default is 'black'.
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
        f_min = min(window)
        f_max = max(window)
        keep = np.where((all_hits_frame["Freq"] >= f_min) & (all_hits_frame["Freq"] <= f_max))
        all_hits_frame = all_hits_frame.iloc[keep]
        
        #keep only the candidates within the window 
        keep = np.where((candidate_event_dataframe["Freq"] > f_min) & (candidate_event_dataframe["Freq"] < f_max))
        candidate_event_dataframe = candidate_event_dataframe.iloc[keep]
        
    #remove hits with a drift rate of zero
    if not check_zero_drift:
        keep = np.where(all_hits_frame["DriftRate"] != 0)
        all_hits_frame = all_hits_frame.iloc[keep]
        
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
            plot_hit_candidate(dat_file_list, 
                               fil_file_list, 
                               source_name_list,
                               all_hits_frame,
                               outdir=outdir,
                               check_zero_drift=check_zero_drift, 
                               alpha=alpha, 
                               color=color,
                               window=window)
    else:
        #plot the hits and candidate(s)
        print("This will make %s .png files"%n_events)
        for i in range(n_events):
            candidate = candidate_event_dataframe.iloc[i]
            plot_hit_candidate(dat_file_list, 
                               fil_file_list, 
                               source_name_list,
                               all_hits_frame,
                               candidate,
                               outdir=outdir,
                               check_zero_drift=check_zero_drift, 
                               alpha=alpha, 
                               color=color,
                               window=window)
        
def plot_hit_candidate(dat_file_list, fil_file_list, source_name_list, all_hits_frame, candidate=None, check_zero_drift=False, 
                 outdir=None, alpha=1, color='black', window=None):
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
    color : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default is 'black'.
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
        f_start = min(window)
        f_stop  = max(window)
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
    print('plot_dat plot_hit_candidate: max_load={} is required for {}'.format(max_load, fil_file_list[0]))
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
        
        this_plot = plot_event.plot_waterfall(wf, 
                                              source_name_list[i], 
                                              f_start, 
                                              f_stop)
        
        make_plot(dat_file_list[i], fil_file_list[i],
                 f_start, f_stop, t0, candidate, check_zero_drift=check_zero_drift, alpha=alpha, color=color)
        
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

    #ax = plt.gca()
    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
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

def make_plot(dat, fil, f_start, f_stop, t0, candidate=None, check_zero_drift=False, alpha=1, color='black'):
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
    color : str, optional
        Allows for the specification of the color of the overlayed
        hits. The default is 'black'.

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
        plot_event.overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration)
    
    if len(hit_frame) == 0:
        # there are no hits detected in this dat file 
        return
    
    if not check_zero_drift:
        hit_frame = hit_frame[hit_frame["DriftRate"] != 0]
    
    #f_mid = 0.5 * (f_start + f_stop)
    t_duration = (wf.n_ints_in_file - 1) * wf.header["tsamp"]
    
    #plot all the hits 
    bandwidth = 500./1e6
    half_bandwidth = bandwidth/2.0
    for i in range(len(hit_frame)):
        hit = hit_frame.iloc[i]
        
        f_mid = hit["Freq"]
        drift_rate = hit["DriftRate"]
        f_event = f_mid 
        start, stop = np.sort((f_mid - (half_bandwidth),  f_mid + (half_bandwidth)))
        
        plot_event.overlay_drift(f_event, start, stop, drift_rate, t_duration, offset=0, alpha=alpha, color=color)
        