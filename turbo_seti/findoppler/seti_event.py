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
    setattr(parser.values, option.dest, value.replace('[','').replace(']','').split(','))

def main():

    p = OptionParser()
    p.set_usage('python seti_event.py <FULL_PATH_TO_FIL_FILE> [options]')

#    p.add_option('-m', '--min_drift', dest='min_drift', type='float', default=0.0, help='Set the minimum drift rate to search. Unit: Hz/sec. Default:0.0')
    p.add_option('-M', '--max_drift', dest='max_drift', type='float', default=10.0, help='Set the drift rate to search. Unit: Hz/sec. Default: 10.0')
    p.add_option('-s', '--snr', dest='snr', type='float', default=25.0, help='SNR threshold. Default: 25.0')
#    p.add_option('-b', '--bw', dest='bw', type='float', default=1, help='Specify the amount of \'compression\' to be done in frequency domain to search for more \'spread out\' signals. Unit:?. Default: ?')
    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='./', help='Location for output files. Default: local dir. ')
#    p.add_option('-w', '--width', dest='slice_width', type='int', default=512, help='')
    p.add_option('-l', '--loglevel', dest='loglevel', type='str', default='info', help='Specify log level (info, debug)')
    p.add_option('-c', '--coarse_chans', dest='coarse_chans', type='str', action='callback', default='',callback=make_list, help='Coma separated list of coarse channels to analyze.(ie. "5,8" to do from 5th to 8th coarse channels)')
    opts, args = p.parse_args(sys.argv[1:])

    if len(args)!=1:
        print('Please specify a file name \nExiting.')
        sys.exit()
    else:
        filename = args[0]

    # Stuff needed for LOFAR version. will remove
    obs_info = {}
    obs_info['pulsar'] = 0  # Bool if pulsar detection.
    obs_info['pulsar_found'] = 0  # Bool if pulsar detection.
    obs_info['pulsar_dm'] = 0.0  # Pulsar expected DM.
    obs_info['pulsar_snr'] = 0.0 # SNR
    obs_info['pulsar_stats'] = np.zeros(6)
    obs_info['RFI_level'] = 0.0
    obs_info['Mean_SEFD'] = 0.0
    obs_info['psrflux_Sens'] = 0.0
    obs_info['SEFDs_val'] = [0.0]
    obs_info['SEFDs_freq'] = [0.0]
    obs_info['SEFDs_freq_up'] = [0.0]

    #Setting log level
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

        find_seti_event = FinDoppler(filename, max_drift = opts.max_drift, snr = opts.snr, out_dir = opts.out_dir,coarse_chans = opts.coarse_chans, obs_info=obs_info)
        find_seti_event.search()
##EE-benshmark    cProfile.runctx('find_seti_event.search()',globals(),locals(),filename='profile_search_M%2.1f_S%2.1f_t%i'%(opts.max_drift,opts.snr,int(os.times()[-1])))

        t1 = time.time()
        print('Search time: %5.2f min' % ((t1-t0)/60.))

    except Exception as e:
        logging.exception(e)
        raise Exception(1,'[turbo_SETI] Some issue with FinDoppler.',e)

if __name__=='__main__':
    main()
