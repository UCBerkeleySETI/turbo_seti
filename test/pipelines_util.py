r'test_pipelines_N.py utilities'

import numpy as np
import pandas as pd


EXP_SOURCE = ['VOYAGER-1'] * 2
EXP_TOPHITNUM = [1, 2]
EXP_STATUS = ['on_table_1', 'on_table_1']
EXP_CHANINDX = [651879, 659989]
SNR_LOW = np.array([21.0, 192.0])
SNR_HIGH = np.array([24.0, 194.0])
CSV_DELIM = ','


def validate_hittbl(arg_pd_df, arg_csvf, arg_caller, arg_n_events):
    r'''
    Read in the CSV file into a raw Pandas DataFrame.
    Check that specific columns have the expected values:
      Source, TopHitNum, status, ChanIndx, and SNR.
    Check that the CSV and the Pandas dataframe match.
    '''
    prefix = arg_caller + '[validate_hittbl]'
    df = pd.read_csv(arg_csvf, sep=CSV_DELIM)
    #df.drop('Unnamed: 0')
    nrows = len(df)
    if nrows != arg_n_events:
        raise ValueError('validate_csvf: Expected {} rows but observed {} rows'
                         .format(arg_n_events, nrows))

    csv_source = df['Source'].tolist()
    csv_tophitnum = df['TopHitNum'].tolist()
    csv_status = df['status'].tolist()
    csv_chanindx = df['ChanIndx'].tolist()
    csv_snr = df['SNR'].tolist()
    if csv_source != EXP_SOURCE:
        raise ValueError('{}: Expected source column {} but observed {}'
                         .format(prefix, EXP_SOURCE, csv_source))
    if csv_tophitnum != EXP_TOPHITNUM:
        raise ValueError('{}: Expected TopHitNum column {} but observed {}'
                         .format(prefix, EXP_TOPHITNUM, csv_tophitnum))
    if csv_status != EXP_STATUS:
        raise ValueError('{}: Expected status column {} but observed {}'
                         .format(prefix, EXP_STATUS, csv_status))
    if csv_chanindx != EXP_CHANINDX:
        raise ValueError('{}: Expected channel index column {} but observed {}'
                         .format(prefix, EXP_CHANINDX, csv_chanindx))
    if np.any(csv_snr > SNR_HIGH) or np.any(csv_snr < SNR_LOW):
        raise ValueError('{}: Expected SNR column in range of {}:{} but observed {}'
                         .format(prefix, SNR_LOW, SNR_HIGH, csv_snr))
    failures = 0

    pd_thn = arg_pd_df['TopHitNum'].values
    if not np.all(csv_tophitnum == pd_thn):
        print('*** Oops, {}: pd_thn={}, csv_tophitnum={}'
              .format(prefix, pd_thn, csv_tophitnum))
        failures += 1

    pd_chanindx = arg_pd_df['ChanIndx'].values
    if not np.all(csv_chanindx == pd_chanindx):
        print('*** Oops, {}: pd_chanindx={}, csv_chanindx={}'
              .format(prefix, pd_chanindx, csv_chanindx))
        failures += 1

    pd_snr = arg_pd_df['SNR'].values
    if not np.all(np.isclose(csv_snr, pd_snr, rtol=0.0001)):
        print('*** Oops, {}: pd_snr={}, csv_snr={}'
              .format(prefix, pd_snr, csv_snr))
        failures += 1

    if failures > 0:
        raise ValueError('{}: CSV and pandas table do not agree!'
                         .format(prefix))
