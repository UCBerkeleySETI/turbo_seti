from turbo_seti.find_doppler.find_doppler import hitsearch, MaxVals
import numpy as np
import logging

#hitsearch(spectrum, specstart, specend, hitthresh, drift_rate, header, tdwidth, max_val, reverse)

s  = np.zeros(1000)
s[100] = 100
s[101] = 100
s0 = 0
s1 = len(s)
hit_thresh = 10
drift_rate = 0
header     =  {'NAXIS1': len(s), 'FCNTR': 0.5, 'DELTAF': 0.001, 'baryv': 0}
maxvals = MaxVals()
maxvals.init(len(s))


logger = logging.getLogger(__name__)

out = hitsearch(s, s0, s1, hit_thresh, drift_rate, header, len(s), maxvals,
                reverse=0, fscrunch=2)