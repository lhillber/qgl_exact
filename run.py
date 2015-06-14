#!/usr/bin/python

import qgl
import multiprocessing as mp



# QGL Simulation
# ==============

# Simulation Parameters
# ---------------------

L_list = [10]
dt_list = [1.0]
tasks = ['t', 'n', 'nn', 'MI']
output_dir = "../output"
t_span_list = [(0.0, 10.0)]
IC_list = [[('a',0.0), ('W',1.0)]]
DATA_PATH = './temp_path'

# Build Simulations
# -----------------

simulations = [ [Simulation(tasks, L, t_span, dt, IC,
                            states_path = DATA_PATH + '/states/',
                            meas_path   = DATA_PATH + '/measures/') \
                         for dt in dt_list \
                         for t_span in t_span_list \
                         for IC in IC_list ] \
                for L in L_list]

# Run Simulations
# ---------------

processes = mp.Pool(12)
processes.map(qgl.evolve_state, simulations)
