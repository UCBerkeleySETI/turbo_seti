#!/usr/bin/env python
'''
Main program module for executable turboSETI
'''

import sys
import os
import logging
import time
import cProfile
import pstats
from argparse import ArgumentParser

from blimpy import __version__ as BLIMPY_VERSION
from .find_doppler import FindDoppler
from .kernels import Kernels
from .turbo_seti_version import TURBO_SETI_VERSION


def main(args=None):
    r"""
    This is the entry-point to turboSETI.

    Parameters
    ----------
    args : dict

    """
    # Create an option parser to get command-line input/arguments
    p = ArgumentParser(description='turboSETI doppler drift narrowband search utility version {}.'
                                    .format(TURBO_SETI_VERSION))

    p.add_argument('filename', type=str, default='', nargs="?",
                   help='Name of filename to open (h5 or fil)')
    p.add_argument('-v', '--version', dest='show_version', default=False, action='store_true',
                   help='show the turbo_seti and blimpy versions and exit')
    p.add_argument('-M', '--max_drift', dest='max_drift', type=float, default=10.0,
                   help='Set the maximum drift rate threshold. Unit: Hz/sec. Default: 10.0')
    p.add_argument('-m', '--min_drift', dest='min_drift', type=float, default=0.00001,
                   help='Set the minimum drift rate threshold. Unit: Hz/sec. Default: 0.00001')
    p.add_argument('-s', '--snr', dest='snr', type=float, default=25.0,
                   help='Set the minimum SNR threshold. Default: 25.0')
    p.add_argument('-o', '--out_dir', dest='out_dir', type=str, default='./',
                   help='Location for output files. Default: local dir. ')
    p.add_argument('-l', '--loglevel', dest='log_level', type=str, default='info',
                   help='Specify log level (info, debug, warning)')
    p.add_argument('-c', '--coarse_chans', dest='coarse_chans', type=str, default=None,
                   help='Comma separated string list of coarse channels to analyze. E.g. 7,12 to search channels 7 and 12 only.')
    p.add_argument('-n', '--n_coarse_chan', dest='n_coarse_chan', type=int, default=None,
                   help='Number of coarse channels to use.')
    p.add_argument('-p', '--n_parallel', dest='n_parallel', type=int, default=1,
                   help='Number of dask partitions to run in parallel. Default to 1 (dask not in use)')
    p.add_argument('-b', '--progress_bar', dest='flag_progress_bar', type=str, default='n',
                   help='Use a progress bar with dask? (y/n)')
    p.add_argument('-g', '--gpu', dest='flag_gpu', type=str, default='n',
                   help='Compute on the GPU? (y/n)')
    p.add_argument('-z', '--blank_dc', dest='flag_blank_dc', type=str, default='y',
                   help='Smooth out the DC spike? (y/n)')
    p.add_argument('-d', '--gpu_id', dest='gpu_id', type=int, default=0,
                   help='Use which GPU device? (0,1,...)')
    p.add_argument('-P', '--profile', dest='flag_profile', type=str, default='n',
                   help='Profile execution? (y/n)')
    p.add_argument('-S', '--single_precision', dest='flag_single_precision', type=str, default='y',
                   help='Use single precision (float32)? (y/n)')
    p.add_argument('-a', '--append_output', dest='flag_append_output', type=str, default='n',
                   help='Append output DAT & LOG files? (y/n)')

    if args is None:
        args = p.parse_args()
    else:
        args = p.parse_args(args)

    if args.show_version:
        print('turbo_seti: {}'.format(TURBO_SETI_VERSION))
        print('blimpy: {}'.format(BLIMPY_VERSION))
        sys.exit(0)

    if args.filename == '':
        os.system('turboSETI -h')
        sys.exit(0)

    if not os.path.exists(args.filename):
        print("\nInput file {} does not exist!\n".format(args.filename))
        sys.exit(86)

    if args.flag_profile == "y":
        cProfile.runctx('exec(args)', {'args': args, 'exec': exec_proc}, {}, filename='stats_file.bin')
        p = pstats.Stats('stats_file.bin')
        p.strip_dirs().sort_stats('time').print_stats(16)
    else:
        exec_proc(args)

def exec_proc(args):
    r"""
    Interface to FindDoppler class, called by main().

    Parameters
    ----------
    args : dict

    """
    if args.coarse_chans is None:
        coarse_chans = ''
    else:
        coarse_chans = map(int, args.coarse_chans.split(','))

    # Setting log level
    if args.log_level == 'info':
        log_level_int = logging.INFO
    elif args.log_level == 'debug':
        log_level_int = logging.DEBUG
    elif args.log_level == 'warning':
        log_level_int = logging.WARNING
    else:
        raise ValueError('Need valid loglevel value (info, debug, warning).')

    fmt = '%(name)-15s %(levelname)-8s %(message)s'
    logging.basicConfig(format=fmt, level=logging.WARNING)

    if Kernels.has_gpu() and args.flag_gpu == "n":
        print("Info: Your system is compatible with GPU-mode. Use the `-g y` argument to enable it.")

    #Doing search
    t0 = time.time()

    find_seti_event = FindDoppler(args.filename,
                                  max_drift=args.max_drift,
                                  min_drift=args.min_drift,
                                  snr=args.snr,
                                  out_dir=args.out_dir,
                                  append_output=(args.flag_append_output == "y"),
                                  coarse_chans=coarse_chans,
                                  obs_info=None,
                                  n_coarse_chan=args.n_coarse_chan,
                                  gpu_backend=(args.flag_gpu == "y"),
                                  gpu_id=args.gpu_id,
                                  blank_dc=(args.flag_blank_dc == "y"),
                                  precision=1 if args.flag_single_precision == "y" else 2,
                                  log_level_int=log_level_int)

    find_seti_event.search(n_partitions=args.n_parallel,
                           progress_bar=args.flag_progress_bar)

    t1 = time.time()
    print('Search time: %5.2f min' % ((t1-t0)/60.))


if __name__ == '__main__':
    main()
