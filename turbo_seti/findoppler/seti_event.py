#!/usr/bin/env python

from .findopp import FinDoppler

import sys
import os
import logging
import numpy as np
import time
from optparse import OptionParser

#For debugging
#import cProfile
#import pdb;# pdb.set_trace()

def make_list(option, opt_str, value, parser):
    """
    This function is used when certain commandline options are used when running turboSETI (-c). It is used to convert
    the inputted option, which should be a comma separated string, into a list. Stores this in the inputted OptionParser
    object's values field, under the inputted option. This is only called for the coarse_chans option currently,
    essentially to parse the inputted coarse channels into a usable format.
    :param option:      Option,         whose value should be set to the new list
    :param opt_str:     string,         (unused) command line option inputted when main was run
    :param value:       string,         in comma separated format which must be converted to list format
    :param parser:      OptionParser,   instance whose option we are setting to the new list
    :return: void
    """
    v = value.replace('[','').replace(']','').split(',')
    v = list(map(int, v))
    setattr(parser.values, option.dest, v)

def main():
    """
    This is the entry-point to turboSETI.
    """
    # Create an option parser to get command-line input/arguments
    p = OptionParser()
    p.set_usage('turboSETI <FULL_PATH_TO_FIL_FILE> [options]')

#    p.add_option('-m', '--min_drift', dest='min_drift', type='float', default=0.0, help='Set the minimum drift rate to search. Unit: Hz/sec. Default:0.0')
    p.add_option('-M', '--max_drift', dest='max_drift', type='float', default=10.0, help='Set the drift rate to search. Unit: Hz/sec. Default: 10.0')
    p.add_option('-s', '--snr', dest='snr', type='float', default=25.0, help='SNR threshold. Default: 25.0')
#    p.add_option('-b', '--bw', dest='bw', type='float', default=1, help='Specify the amount of \'compression\' to be done in frequency domain to search for more \'spread out\' signals. Unit:?. Default: ?')
    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='./', help='Location for output files. Default: local dir. ')
#    p.add_option('-w', '--width', dest='slice_width', type='int', default=512, help='')
    p.add_option('-l', '--loglevel', dest='loglevel', type='str', default='info', help='Specify log level (info, debug)')
    p.add_option('-c', '--coarse_chans', dest='coarse_chans', type='str', action='callback', default='', callback=make_list,
                 help='Comma separated list of coarse channels to analyze.(ie. "5,8" to do from 5th to 8th coarse channels)')
    p.add_option('-n', '--n_coarse_chan', dest='n_coarse_chan', type=int, default=None,
                 help='Number of coarse channels in file.')
    opts, args = p.parse_args(sys.argv[1:])

    # Makes sure exactly one file is given as an arg
    if len(args) != 1:
        print('Please specify a file name \nExiting.')
        sys.exit()
    else:
        filename = args[0]

    # Stuff needed for LOFAR version. will remove
    obs_info = {}
    obs_info['pulsar'] = 0  # Bool if pulsar detection.
    obs_info['pulsar_found'] = 0  # Bool if pulsar detection.
    obs_info['pulsar_dm'] = 0.0  # Pulsar expected DM.
    obs_info['pulsar_snr'] = 0.0  # Signal toNoise Ratio (SNR)
    obs_info['pulsar_stats'] = np.zeros(6)
    obs_info['RFI_level'] = 0.0  # Radio Frequency Interference
    obs_info['Mean_SEFD'] = 0.0  # Mean System Equivalent Flux Density
    obs_info['psrflux_Sens'] = 0.0
    obs_info['SEFDs_val'] = [0.0]  # System Equivalent Flux Density values
    obs_info['SEFDs_freq'] = [0.0]  # System Equivalent Flux Density frequency
    obs_info['SEFDs_freq_up'] = [0.0]

    # Setting log level
    if opts.loglevel == 'info':
        level_log = logging.INFO
    elif opts.loglevel == 'debug':
        level_log = logging.DEBUG
    else:
        raise ValueError('Need valid loglevel value (info,debug).')

    if level_log == logging.INFO:
        stream = sys.stdout
        format = '%(name)-15s %(levelname)-8s %(message)s'
    else:
        stream =  sys.stderr
        format = '%%(relativeCreated)5d (name)-15s %(levelname)-8s %(message)s'

    #Doing search
    try:
        t0 = time.time()

        logging.basicConfig(format=format,stream=stream,level = level_log)

        find_seti_event = FinDoppler(filename, max_drift=opts.max_drift, snr=opts.snr, out_dir=opts.out_dir,
                                     coarse_chans=opts.coarse_chans, obs_info=obs_info, n_coarse_chan=opts.n_coarse_chan)
        find_seti_event.search()
##EE-benshmark    cProfile.runctx('find_seti_event.search()',globals(),locals(),filename='profile_search_M%2.1f_S%2.1f_t%i'%(opts.max_drift,opts.snr,int(os.times()[-1])))

        t1 = time.time()
        print('Search time: %5.2f min' % ((t1-t0)/60.))

    except Exception as e:
        logging.exception(e)
        raise Exception(1,'[turbo_SETI] Some issue with FinDoppler.',e)

if __name__=='__main__':
    main()
