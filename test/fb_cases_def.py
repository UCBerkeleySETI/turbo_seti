r'''
Data definitions for test_fb_cases.py
'''

import os
from tempfile import gettempdir
from astropy import units as u

#---------- Constants ------------
TESTDIR = gettempdir() + '/test_fb_cases/'
PATH_FIL_FILE = TESTDIR + 'abc.fil'
MIN_SNR = 50
DEBUGGING = False
GENREF = False
HERE = os.path.split(os.path.abspath(__file__))[0]
THE_MEANING_OF_LIFE = 42

class TestResultRecord:
    r'''
    Object definition for a test result record
    '''


    def __init__(self):
        self.fdir = 0 # frequency direction: +1 is ascending, -1 is descending
        self.drsign = 0 # drift rate algebraic sign: +1 is positive, -1 is negative
        self.tophit_id = 0 # identifier of this top hit (1 or 2)
        self.drate = 0.0 # drift rate of this top hit
        self.snr = 0.0
        self.freq = 0.0 # corrected frequency
        self.index = 0 # index to the frequencies


    def to_string(self):
        '''
        Return a displayable string of attribute values.
        '''
        return 'fdir: ' + str(self.fdir) + ', drsign: ' + str(self.drsign) \
                + ', tophit_id: ' + str(self.tophit_id) + ', drate: ' + str(self.drate) \
                + ', snr: ' + str(self.snr) + ', freq: ' + str(self.freq) \
                + ', index: ' + str(self.index)


class SetigenParms:


    def __init__(self):

        # Parameters for all hits
        self.fchans = 1048576 # number of (fine) channels
        self.tchans = 60 # number of time samples
        self.df = 1.0 * u.Hz # fine channel width in Hz
        self.dt = 1.0 * u.s #sampling time in seconds
        self.fch1 = 8421.386717353016 * u.MHz # Starting frequency in MHz

        # Signal 1 parameters
        self.signal_start_1 = self.fchans / 5 # index to frequency columns
        self.drift_rate_1 = 1.3 * u.Hz/u.s # drift rate to inject
        self.width_1 = 1.0 * u.Hz # signal width in Hz
        self.snr_1 = MIN_SNR + 50 # SNR which will determine setigen intensity level

        # Signal  2 parameters
        self.signal_start_2 = 4 * self.fchans / 5
        self.drift_rate_2 = 2.6 * u.Hz/u.s
        self.width_2 = 2.0 * u.Hz
        self.snr_2 = MIN_SNR * 6.0

        # Signal 3 parameters
        self.signal_start_3 = self.fchans / 3
        self.drift_rate_3 = 3.9 * u.Hz/u.s
        self.width_3 = 40.0 * u.Hz
        self.snr_3 = MIN_SNR - 20.0
        if self.snr_3 < 0:
            self.snr_3 = 0.001

        # Signal 4 parameters
        self.signal_start_4 = 9 * self.fchans / 16
        self.drift_rate_4 = 100.0 * u.Hz/u.s
        self.width_4 = 40.0 * u.Hz
        self.snr_4 = MIN_SNR + 20.0

        # Signal 5 is similar to signal 4 but drifting in the opposite direction.
        self.signal_start_5 = 7 * self.fchans / 16
        self.drift_rate_5 = -self.drift_rate_4 # opposite direction as signal 4
        self.width_5 = self.width_4
        self.snr_5 = self.snr_4
