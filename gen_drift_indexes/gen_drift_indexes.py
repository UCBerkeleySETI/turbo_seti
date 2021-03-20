r''' Create a drift index file. '''

from argparse import ArgumentParser

TRACING = False


def generate_row(arg_ncols):
    r''' Generate a row template. '''
    wrow = [0]
    wrowlen = 1
    for ii in range(arg_ncols):
        if ii % 2 == 1:
            wrow.append(ii)
            wrowlen += 1
    padlen = arg_ncols - wrowlen
    for ii in range(padlen):
        wrow.append(0)
    return wrow


def elem_insert(arg_row, arg_elem):
    r''' Insert one element into the current row. '''
    if TRACING:
        print('TRACE elem_insert row={}, elem={}'.format(arg_row, arg_elem))
    # Check for duplicate.
    if arg_elem in arg_row:
        if TRACING:
            print('TRACE elem_insert ignoring dup')
        return
    # Get index of neighbour to the right.
    windex = arg_row.index(arg_elem + 1)
    # Insert just to left of neighbour.
    arg_row.insert(windex, arg_elem)
    # If the last element is 0, discard it.
    if arg_row[-1] == 0:
        arg_row.pop(-1)


def writer(arg_fh, arg_row):
    r''' Write the text of the given row to the output file. '''
    buffer = ''
    for elem in arg_row:
        buffer += ' ' + str(elem)
    arg_fh.write(buffer + '\n')


def proc_one_file(arg_dirpath, arg_filenumber):
    r''' Generate one drift index file. '''

    filepath = arg_dirpath + '/' + 'drift_indexes_array_{}.txt'.format(arg_filenumber)
    nrows = 2 ** (arg_filenumber - 1)
    ncols = 2 ** arg_filenumber
    print('Begin {}, nrows={}, ncols={}'.format(filepath, nrows, ncols))

    with open(filepath, "w") as fh:

        # Write the first row (#0).
        row_0 = generate_row(ncols)
        if TRACING:
            print('TRACE ii=1')
        writer(fh, row_0)

        if TRACING:
            print('TRACE begin row-loop')

        # Construct and write the remaining rows.
        for row_num in range(1, nrows):

            if TRACING:
                print('TRACE row_num={}'.format(row_num))

            # If this is an odd row number, insert nrows into to the row as an element.
            row = generate_row(ncols)
            if row_num % 2 == 1: # odd row number?
                elem_insert(row, nrows)

            # For each power of 2 (2, 4, 8, ..., 2^(arg_filenumber - 1)),
            #    Create a divisor = that power of 2.
            #    Use the divisor as a selection mechanism for the current row:
            #       If (row_num // divisor) is odd,
            #       then insert all of the elements from the range
            #           nrows // divisor up to ncols by 2 * (nrows // divisor).
            for jj in range(1, arg_filenumber):
                divisor = 2 ** jj
                if (row_num // divisor) % 2 == 1:
                    start_ix = nrows // divisor
                    for elem in range(start_ix, ncols, 2 * start_ix):
                        elem_insert(row, elem)

            # Write the accumulated current row.
            writer(fh, row)

    print('End {}'.format(filepath))


def cmd_tool(args=None):
    r''' Main program of the command-line tool. '''
    p = ArgumentParser(description='Generate a drift index file.')
    p.add_argument('dirpath', type=str,
                   help='Path of directory to write the drift index file.')
    p.add_argument('-n', '--file-number', type=int, default=11, dest='max_file_number',
                   help='Maximum value for the log base 2 of the number of time integration steps.')
    if args is None:
        args = p.parse_args()
    else:
        args = p.parse_args(args)
    
    for fn in range(2, args.max_file_number + 1):
        proc_one_file(args.dirpath, fn)


if __name__ == "__main__":
    cmd_tool()
