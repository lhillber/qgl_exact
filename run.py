#! /usr/bin/python

import qgl


# Simulation Parameters
# =====================

L_list = [10]
dt_list = [1.0]
tasks = ['t', 'n', 'nn', 'MI']
output_dir = "../output"
t_span_list = [(0.0, 10.0)]
IC_list = [[('a',0.0), ('W',1.0)]]


# Run Simulation
# ==============

for L in L_list:
    for dt in dt_list:
        for t_span in t_span_list:
            for IC in IC_list:

                qgl.run_sim (L = L, dt = dt, t_span = t_span, IC = IC, 
                        output_dir = output_dir, tasks = tasks)
