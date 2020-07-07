#!/usr/bin/env python

from .find_doppler import FindDoppler

import sys
import os
import logging
import numpy as np
import time
from argparse import ArgumentParser


def main(args=None):
    """This is the entry-point to turboSETI.

    Args:
      args: (Default value = None)

    Returns:

    """
    # Create an option parser to get command-line input/arguments
    p = ArgumentParser(description='turboSETI doppler drift narrowband search utility.')

    p.add_argument('filename', type=str, help='Name of filename to open (h5 or fil)')
    p.add_argument('-M', '--max_drift', dest='max_drift', type=float, default=10.0,
                   help='Set the drift rate to search. Unit: Hz/sec. Default: 10.0')
    p.add_argument('-s', '--snr', dest='snr', type=float, default=25.0,
                   help='SNR threshold. Default: 25.0')
    p.add_argument('-o', '--out_dir', dest='out_dir', type=str, default='./',
                   help='Location for output files. Default: local dir. ')
    p.add_argument('-l', '--loglevel', dest='loglevel', type=str, default='info',
                   help='Specify log level (info, debug)')
    p.add_argument('-c', '--coarse_chans', dest='coarse_chans', type=str, default=None,
                   help='Comma separated list of coarse channels to analyze.')
    p.add_argument('-n', '--n_coarse_chan', dest='n_coarse_chan', type=int, default=None,
                   help='Number of coarse channels in file.')

    if args is None:
        args = p.parse_args()
    else:
        args = p.parse_args(args)

    if args.coarse_chans is None:
        coarse_chans=''
    else:
        coarse_chans = map(int, args.coarse_chans.split(',') )

    # Setting log level
    if args.loglevel == 'info':
        level_log = logging.INFO
    elif args.loglevel == 'debug':
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

        find_seti_event = FindDoppler(args.filename, max_drift=args.max_drift, snr=args.snr, out_dir=args.out_dir,
                                      coarse_chans=coarse_chans, obs_info=None, n_coarse_chan=args.n_coarse_chan)
        find_seti_event.search()

        t1 = time.time()
        print('Search time: %5.2f min' % ((t1-t0)/60.))

    except Exception as e:
        logging.exception(e)
        raise Exception(1,'[turbo_SETI] Some issue with FindDoppler.',e)

if __name__=='__main__':
    main()
