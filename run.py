#!/usr/bin/python3
import qgl
import qgl_util
import qgl_plotting
from math import sin, cos, pi
import numpy as np

# QGL Simulation
# ==============

# Simulation Parameters
# ---------------------

post_processing = True

th_list = np.linspace(0.0, pi/4, 25)
L_list      = [5,8,11]
dt_list     = [0.01]
tasks       = ['t', 'n', 'nn', 'MI']
t_span_list = [(100, 120)]
IC_list     = [[('a', cos(th)), ('W',sin(th))] for th in th_list]
#IC_list = [[('s3',1.0)]]
output_dir  = "avW5_8_11"

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
if not post_processing:
    qgl_util.multi_runs(qgl_util.run_sims, simulations)


# Post Processing
# ===============
if post_processing:
    qgl_plotting.main(output_dir, L_list, dt_list, \
        t_span_list, IC_list, th_list)
