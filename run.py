#! /usr/bin/python

import qgl


# Simulation Parameters
# =====================

L_list = [10]
dt_list = [1.0]
tasks = ['t', 'n', 'nn', 'MI']
output_dir = "output"
t_span_list = [(0.0, 10.0), (2.0, 12.0)]
IC_list = [[('a',1.0), ('W',0.0)]]


# Run Simulation
# ==============

qgl.run_sim (L_list = L_list, dt_list = dt_list, output_dir = output_dir,
                tasks = tasks, t_span_list = t_span_list, IC_list = IC_list,
                split_processes = "L")
