#!/usr/bin/env python

from os import chdir, getcwd, listdir, remove, rename
from argparse import ArgumentParser

PREFIX = "_c_"
DEBUGGING = False

def merge_dats_logs(arg_h5: str, arg_dir: str, arg_type: str, cleanup='n'):
    r"""
    Combine DAT or LOG files.

    Parameters
    ----------
    arg_h5 : str
        HDF5 file used by :func:`~turbo_seti.find_doppler.find_doppler.FindDoppler.search`
        to produce the DAT and LOG files.
    arg_dir : str
        Directory holding multiple DAT and LOG files after FindDoppler.search()
        which ran with more than 1 partition.
    arg_type : str
        File extension of interest ('dat' or 'log').

    """
    print("merge_dats_logs: dir={}, type={}, cleanup={}"
          .format(arg_dir, arg_type, cleanup))
    RETURN_TO = getcwd() # Save our current directory path
    chdir(arg_dir) # Change directory
    suffix = '.' + arg_type # E.g. .dat
    files = []
    filenamestem = arg_h5.split('/')[-1].replace('.h5', '')
    len_filenamestem = len(filenamestem)
    print('merge_dats_logs: Working on filename-stem {} type {}'
          .format(filenamestem, arg_type))
    sorted_file_list = sorted(listdir(arg_dir))
    counter = 0
    if DEBUGGING:
        print("DEBUG merge_dats_logs: listdir=", sorted_file_list)
    for cur_file in sorted_file_list:
        cur_type = cur_file.split('.')[-1]
        if cur_type == arg_type and not cur_file.startswith(PREFIX):
            # This is the type of file we are looking for.
            # and it is not the combination version we are building.
            # Does cur_file match the HDF5 file?
            if cur_file[0:len_filenamestem] == filenamestem:
                files.append(cur_file)
                if DEBUGGING:
                    print("DEBUG merge_dats_logs: Selected for merging: ", cur_file)
                counter += 1
    if counter < 1:
        print("*** merge_dats_logs: Nothing selected for merging")
        chdir(RETURN_TO)
        return

    # Append the combo file with each list member.
    path_prefixed_combo = PREFIX + filenamestem + suffix
    with open(path_prefixed_combo, "w") as outfile:
        # Write first file encountered fully.
        with open(files[0], "r") as fd:
            for line in fd:
                outfile.write(line)
        # Write subsequent files, filtering out comment lines (start with '#')
        for cur_file in files[1:]:
            with open(cur_file, "r") as fd:
                for line in fd:
                    if not line.startswith("#"):
                        outfile.write(line)
                        
    # if cleanup is requested, do it now.
    if cleanup == 'y':
        # Remove all of the partitions.
        for cur_file in files:
            remove(cur_file)        
            if DEBUGGING:
                print("merge_dats_logs: Removed: ", cur_file)
        # Rename the merged file
        path_merge_file = filenamestem + suffix
        try:
            rename(path_prefixed_combo, path_merge_file)
            print("merge_dats_logs: Merged into", path_merge_file)
        except Exception as exc:
            print("*** os.rename({}, {}) failed, reason:{}\n"
                  .format(path_prefixed_combo, path_merge_file, str(exc)))

    # Change back to caller's current directory
    chdir(RETURN_TO)

def main(args=None):
    r"""
    Utility for combining multiple DAT and LOG files after a turboSETI run.
    Main procedure for cleaning up after turboSETI.
    
    Parameters
    ----------
    args : dict
        Command line parameters.

    Version 1:
    - Karen Perez
    - Richard Elkins

    """
    p = ArgumentParser(description='Merge DATs & LOGs after turboSETI.')
    p.add_argument('h5_file', type=str, help='Path of HDF5 file used to create DAT & LOG files.')
    p.add_argument('directory', type=str, help='Path of directory containing the DAT & LOG files.')
    p.add_argument('-c', '--cleanup', dest='flag_cleanup', type=str, default='n',
                   help='Cleanup after merging DAT and LOG files? (y/n)')
    
    if args is None:
        args = p.parse_args()
    else:
        args = p.parse_args(args)

    merge_dats_logs(args.h5_file, args.directory, 'dat', cleanup=args.flag_cleanup)
    merge_dats_logs(args.h5_file, args.directory, 'log', cleanup=args.flag_cleanup)

if __name__ == '__main__':
    main()
