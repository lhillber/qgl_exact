#!/usr/bin/python
from math import sqrt
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
from multiprocessing import Process, Pipe
import json

def read_in(meas_file):
    with open(meas_file, 'r') as infile:
       data =  json.load(infile)
    return data

data = read_in('../david_opop/L6_dt1.0_t_span0-10_ICa1.000.meas')

ninj_time_series = [data['nn'][t]['net'] for t in range(10)]
ni_time_series = [data['n'][t]['nexp'] for t in range(10)]

EC_time_series = data['EC']

ND_time_series= [data['MI'][t]['ND'] for t in range(10)]
Y_time_series= [data['MI'][t]['Y'] for t in range(10)]
CC_time_series= [data['MI'][t]['CC'] for t in range(10)]
IHL_time_series= [data['MI'][t]['HL'] for t in range(10)]
EV_time_series= [data['MI'][t]['EV'] for t in range(10)]


times = data['t']

def boundary_terms_gen(self, L):
    L_terms = [
              ['mix', 'n', 'n', 'I'] + ['I']*(L-4),   \
              ['n', 'mix', 'n', 'n'] + ['I']*(L-4),   \
              ['nbar', 'mix', 'n', 'n'] + ['I']*(L-4),\
              ['n', 'mix', 'nbar', 'n'] + ['I']*(L-4),\
              ['n', 'mix', 'n', 'nbar'] + ['I']*(L-4) \
                                      ] 
    R_terms = list(map(lambda L_term: L_term[::-1], L_terms))
    boundary_terms = L_terms+R_terms
    return boundary_terms

