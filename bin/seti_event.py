#!/usr/bin/env python

import sys
import os
import logging
import dedoppler_bones
import numpy as np

def main():
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('python seti_event.py <FULL_PATH_TO_FIL_FILE> [options]')

#    p.add_option('-m', '--min_drift', dest='min_drift', type='float', default=0.0, help='Set the minimum drift rate to search. Unit: Hz/sec. Default:0.0')
    p.add_option('-M', '--max_drift', dest='max_drift', type='float', default=10.0, help='Set the drift rate to search. Unit: Hz/sec. Default: 10.0')
    p.add_option('-s', '--snr', dest='snr', type='float', default=25.0, help='SNR threshold. Default: 25.0')
#    p.add_option('-b', '--bw', dest='bw', type='float', default=1, help='Specify the amount of \'compression\' to be done in frequency domain to search for more \'spread out\' signals. Unit:?. Default: ?')
    p.add_option('-r', '--rfithresh', dest='rfithresh', type='float', default=1000.0, help='Specify the RFI threshold. Default: 1000.0')
    p.add_option('-p', '--path', dest='out_dir', type='str', default='/tmp', help='In the case that the input file size is too big to handle at once, we\'ll need to split it into smaller FITS files. This option specify where to put those FITS files. Default: /tmp ')
#    p.add_option('-w', '--width', dest='slice_width', type='int', default=512, help='')
    p.add_option('-l', '--loglevel', dest='loglevel', type='str', default='info', help='Specify log level (info, debug)')

    opts, args = p.parse_args(sys.argv[1:])

    if len(args)!=1:
        print 'Please specify a FITS file \nExiting.'
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
        logging.basicConfig(format=format,stream=stream,level = level_log)

        mydedopp = dedoppler_bones.dedopp.DedopplerTask(filename, max_drift = opts.max_drift, snr = opts.snr, rfithresh = opts.rfithresh, split_dir = opts.out_dir, obs_info=obs_info, LOFAR=False)
        mydedopp.search()


    except Exception as e:
        logging.exception(e)
        raise Exception(1,'[turbo_SETI] Some issue with dedoppler.',e)

if __name__=='__main__':
    main()
