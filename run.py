#!/usr/bin/python

import qgl


# QGL Simulation
# ==============

# Simulation Parameters
# ---------------------

L_list      = [10]
dt_list     = [1.0]
tasks       = ['t', 'n', 'nn', 'MI']
t_span_list = [(0.0, 10.0)]
IC_list     = [[('a',0.0), ('W',1.0)]]
output_dir  = "../output"

# Simulations to Run
# ------------------

simulations = [ [Simulation (tasks = tasks,  L  = L,
                            t_span = t_span, dt = dt,
                            IC     = IC,     output_dir = output_dir) \
                     for dt     in dt_list \
                     for t_span in t_span_list \
                     for IC     in IC_list] \
                for L in L_list]

# Run them!
# ---------

qgl.run_sims(simulations)
