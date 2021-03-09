r''' Source file for merge_dats_logs() '''

from os import chdir, getcwd, listdir, remove, rename

PREFIX = "_c_"
DEBUGGING = False
DATLNFMT = '{:>4s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>9s}  {:>12s}  {:>12s}' \
    + '{:>4s}  {:>12s}  {:>5s}  {:>9s}\n'

def merge_dats_logs(arg_h5: str, arg_dir: str, arg_type: str, cleanup='n'):
    r"""
    Merge multiple DAT (or LOG) files.

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
        if arg_type == 'dat':
            tophit_counter = 0
        for cur_file in files[1:]:
            with open(cur_file, "r") as fd:
                for inline in fd:
                    if not inline.startswith("#"): # not a comment
                        if arg_type == 'dat': # renumber tophit number field
                            tophit_counter += 1
                            outlist = inline.split()
                            if DEBUGGING:
                                print('DEBUG outlst:', outlist)
                            outlist[0] = str(tophit_counter)
                            outfile.write(DATLNFMT.format(*outlist))
                        else: # log file
                            outfile.write(inline)

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
