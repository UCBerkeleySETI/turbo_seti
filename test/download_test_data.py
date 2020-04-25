import os

import h5py
import hdf5plugin

HERE = os.path.split(os.path.abspath(__file__))[0]

def download_voyager_data():
    """ Download Voyager test data """
    # TODO: move Voyager data out of repo -- currently, this call is not needed as repo comes with data.
    pass


def flip_data(filename):
    """ Flip Voyager data along frequency axis.

    The flipped file is used to check logic works when frequency is inverted.
    """
    print("Generating frequency flipped version of Voyager data...")
    assert filename.endswith('.h5')
    flipped_filename = filename.replace('.h5', '.flipped.h5')
    os.system('cp %s %s' % (filename, flipped_filename))
    with h5py.File(flipped_filename, 'r+') as h:
        foff_orig = h['data'].attrs['foff']
        fch1_orig = h['data'].attrs['fch1']
        nchans    = h['data'].attrs['nchans']
        fchN      = fch1_orig + (foff_orig * nchans)
        h['data'].attrs['foff'] = foff_orig * -1
        h['data'].attrs['fch1'] = fchN

        for ii in range(h['data'].shape[0]):
            print('\tFlipping %i/%i' % (ii+1, h['data'].shape[0]))
            h['data'][ii, 0, :] = h['data'][ii, 0][::-1]
    print("Done.")

if __name__ == "__main__":
    voyager_fp = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.h5')
    flip_data(voyager_fp)