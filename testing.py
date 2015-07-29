#!/usr/bin/python
from math import sqrt
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
from multiprocessing import Process, Pipe

import measures


meas = measures.Measurements(['dum'], 'dum')

data = meas.read_in('../david_opop/L6_dt1.0_t_span0-100_ICa1.000.meas')

ninj_time_series = [data['nn'][t]['net'] for t in range(100)]
ni_time_series = [data['n'][t]['nexp'] for t in range(100)]

EC_time_series = data['EC']

ND_time_series= [data['MI'][t]['ND'] for t in range(100)]
Y_time_series= [data['MI'][t]['Y'] for t in range(100)]
CC_time_series= [data['MI'][t]['CC'] for t in range(100)]
HL_time_series= [data['MI'][t]['HL'] for t in range(100)]

times = data['t']

print(HL_time_series) 

