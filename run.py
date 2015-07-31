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

th_list = np.linspace(0.0, pi/4, 25)

L_list      = [6]
dt_list     = [1.0]
tasks       = ['t', 'EC', 'n', 'nn', 'MI']
t_span_list = [(0, 10)]
#IC_list     = [[('c3E0_1, cos(th)), ('c2t0_1', sin(th))] for th in th_list]
IC_list = [[('d57', 1.0)]]
output_dir  = "david_opop"


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
