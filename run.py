#!/usr/bin/python3

import qgl
import qgl_util
import qgl_plotting
from math import sin, cos, pi
import numpy as np
from mpi4py import MPI


# QGL Simulation
# ==============

# Simulation Parameters
# ---------------------

post_processing = False


L_list      = [10]
dt_list     = [0.1/2**k for k in [0,1,2,3,4]]
t_span_list = [(0, 2)]
IC_list = [[('c2d3', 1.0)]]
output_dir  = 'exact_for_trotter'

tasks       = ['t', 'EC', 'n', 'nn', 'MI']


# Simulations to Run
# ------------------
simulations = [ (tasks,  L, t_span, dt, IC, output_dir)
                     for dt     in dt_list
                     for t_span in t_span_list
                     for IC     in IC_list
                     for L      in L_list]

# Run them
# --------
if not post_processing:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, simulation in enumerate(simulations):
        if i % nprocs == rank:
            sim = qgl.Simulation (*simulation)
            del sim

# Post Processing
# ===============
if post_processing:
    qgl_plotting.main(output_dir, L_list, dt_list, \
        t_span_list, IC_list, th_list)
