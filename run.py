#!/usr/bin/python

import qgl
import qgl_util
import qglplotting
from math import sin, cos, pi

# QGL Simulation
# ==============

# Simulation Parameters
# ---------------------

th_list = [0.0, pi/2 ]

L_list      = [7,8]
dt_list     = [0.1]
tasks       = ['t', 'n', 'nn', 'MI']
t_span_list = [(0.0, 4.0)]
IC_list     = [[('a', cos(th)), ('W',sin(th))] for th in th_list]
output_dir  = "multiproc"

# Simulations to Run
# ------------------

simulations = [ [qgl.Simulation (tasks = tasks,  L = L,
                            t_span = t_span, dt = dt,
                            IC     = IC,     output_dir = output_dir)
                     for dt     in dt_list
                     for t_span in t_span_list
                     for IC     in IC_list]
                for L in L_list]

# Run them!
# ---------

qgl_util.multi_runs(qgl_util.run_sims,   simulations)



# Post Processing
# ===============

#qglplotting.main(output_dir, L_list, dt_list, \
#        t_span_list, IC_list, th_list)


