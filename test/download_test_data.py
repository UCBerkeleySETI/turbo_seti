import os
import sys

import h5py
import hdf5plugin

HERE = os.path.split(os.path.abspath(__file__))[0]
MIN_SIZE = 40000000

URL1 = "http://blpd0.ssl.berkeley.edu/Voyager_data/"
FILE1 = "Voyager1.single_coarse.fine_res.h5"

URL2 = "http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel/"
FILE2 = "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"


def oops(msg):
    print("\n*** OOPS, {} !!!\n".format(msg))
    sys.exit(86)
    

def check_h5(path):
    
    try:
        h5 = h5py.File(path, mode='r')
    except:
        oops("Is {} really an HDF5 file?  Probably not".format(path))
    h5.close()


def download_test_data():
    """ Download Voyager test data """

    try:
        os.system('rm *.h5 *.fil *.dat *.log *.png 2> /dev/null')
    except:
        pass
    print("Begin downloads .....\n")

    exit_status = os.system("curl --url '{}/{}'  -o ./{}".format(URL2, FILE2, FILE2))
    if exit_status != 0:
        oops("cannot download {}".format(FILE2))
    sz = os.path.getsize(FILE2)
    if sz < MIN_SIZE:
        oops("Downloaded file {} is way too small, size={}".format(FILE2, sz))
    check_h5(FILE2)

    exit_status = os.system("curl --url '{}/{}'  -o ./{}".format(URL1, FILE1, FILE1))
    if exit_status != 0:
        oops("cannot download {}".format(FILE1))
    sz = os.path.getsize(FILE1)
    if sz < MIN_SIZE:
       oops("Downloaded file {} is way too small, size={}".format(FILE1, sz))
    check_h5(FILE1)
        
    print("\nDownloads ok.")


def create_fil_from_h5(path):
    """ Create a .fil file from an .h5 file. """
    exit_status = os.system("h52fil %s" % path)
    if exit_status != 0:
        oops("h52fil  {}  FAILED".format(path))
    print("h52fil ok.")


def flip_data(filename):
    """ Flip Voyager data along frequency axis.

    The flipped file is used to check logic works when frequency is inverted.
    """
    print("Generating frequency flipped version of Voyager data...")
    assert filename.endswith('.h5')
    flipped_filename = filename.replace('.h5', '.flipped.h5')
    exit_status = os.system('cp %s %s' % (filename, flipped_filename))
    if exit_status != 0:
        oops("cp {} to {} FAILED".format(filename, flipped_filename))
    with h5py.File(flipped_filename, 'r+') as h:
        foff_orig = h['data'].attrs['foff']
        fch1_orig = h['data'].attrs['fch1']
        nchans    = h['data'].attrs['nchans']
        fchN      = fch1_orig + (foff_orig * nchans)
        h['data'].attrs['foff'] = foff_orig * -1
        h['data'].attrs['fch1'] = fchN
        h['data'].attrs['source_name'] = 'Voyager1Flipped'

        for ii in range(h['data'].shape[0]):
            print('\tFlipping %i/%i' % (ii+1, h['data'].shape[0]))
            h['data'][ii, 0, :] = h['data'][ii, 0][::-1]
    print("Done.")


if __name__ == "__main__":
    download_test_data()
    voyager_full_path = os.path.join(HERE, FILE1)
    flip_data(voyager_full_path)
    create_fil_from_h5(voyager_full_path)
