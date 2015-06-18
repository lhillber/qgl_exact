#!/usr/bin/python

from multiprocessing import Pipe
from multiprocessing import Process

from os import makedirs
from os.path import isfile

import time as lap

import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse.linalg as spsla

import qglopsfuncs as qof
import qgl_util as qutil
import qglio as qio
import measures as qms


#==============================================================================
# Model class:
# calculates Hamiltonian and propagator for some L (number of sites) and dt
# (time step)
#==============================================================================
class Model:

    # Build a model
    # -------------
    def __init__(self, L, dt, IC,
                    model_dir = '~/Documents/qgl_ediag'):
        self.L  = L
        self.dt = dt

        self.curr_state = IC
        self.state_list = []
        self.measures   = []

        self.ham_name = 'L'+str(self.L)+'_qgl_ham.mtx'
        self.prop_name = 'L{}_dt{}'.format(self.L, self.dt)+'_qgl_prop'
        self.ham_path  = model_dir + 'hamiltonians/'+self.ham_name
        self.prop_path = model_dir + 'propogators/'+self.prop_name

        self.gen_model ()

        return

    def N3(self,k):
        """
        k:
            site number
        returns the N3 operator of the QGL model at site k
        """
        n3=0
        for  tup in qof.ops('permutations_3'):
            local_matlist3 = [tup[0],tup[1],'mix',tup[2],tup[3]]
            if k==0:
                del local_matlist3[0]
                del local_matlist3[0]
            if k==self.L-1:
                del local_matlist3[3]
                del local_matlist3[3]

            if k==1:
                del local_matlist3[0]
            if k==self.L-2:
                del local_matlist3[3]
            matlist3 = ['I']*(k-2)+local_matlist3
            matlist3 = matlist3 +['I']*(self.L-len(matlist3))
            matlist3 = [qof.ops(key) for key in matlist3]
            n3 = n3 + qof.spmatkron(matlist3)
        return n3


    def N2(self,k):
        """
        k:
            site number
        returns the N2 operator of the QGL model at site k
        """
        n2 = 0
        for tup in qof.ops('permutations_2'):
            local_matlist2 = [tup[0],tup[1],'mix',tup[2],tup[3]]
            if k==0:
                del local_matlist2[0]
                del local_matlist2[0]
            if k==self.L-1:
                del local_matlist2[3]
                del local_matlist2[3]

            if k==1:
                del local_matlist2[0]
            if k==self.L-2:
                del local_matlist2[3]
            matlist2 = ['I']*(k-2)+local_matlist2
            matlist2 = matlist2+['I']*(self.L-len(matlist2))
            matlist2 = [qof.ops(key) for key in matlist2]
            n2 = n2 + qof.spmatkron(matlist2)
        return n2

    def gen_model (self):
        # Hamiltonian
        if isfile(self.ham_path):
            print('Imporiting Hamiltonian...')
            H = sio.mmread(self.ham_path).tocsc()
        else:
            print('Building Hamiltonian...')
            H = sum ([(self.N2(k) + self.N3(k)) for k in range(self.L)])
        self.ham = np.array(H)

        # Propogator
        if isfile(self.prop_path):
            print('Importing Propagator...')
            U0 = np.fromfile (self.prop_path, dtype=complex)
        else:
            print('Building Propagator...')
            U0 = spsla.expm(-1j*self.dt*H).todense().tolist()
        self.prop = np.asarray(U0)


    def write_out (self):
        sio.mmwrite(self.ham_path,self.ham)
        self.prop.tofile(self.prop_path)

    # Generate states up to nmax
    # --------------------------
    def time_evolve (self, nmax):
        new_states = [self.curr_state] * (nmax-len(self.state_list))

        for i in range(1,len(new_states)):
            new_states[i] = self.prop.dot(new_states[i-1])

        self.state_list += new_states
        self.curr_state = self.state_list[-1]


#==============================================================================
# Simulation class
# run time evolution on an initial state and save the result out
#==============================================================================

class Simulation():
    def __init__ (self, tasks, L, t_span, dt, IC, output_dir,
                    model_dir = '~/Documents/qgl_ediag/'):

        makedirs(output_dir, exist_ok=True)

        self.tasks = tasks
        self.L = L
        self.dt = dt
        self.nmin = round(t_span[0]/self.dt)
        self.nmax = round(t_span[1]/self.dt)
        self.nsteps = self.nmax - self.nmin

        self.IC = qutil.make_state(self.L, IC)

        self.model = Model (L, dt, self.IC, model_dir = model_dir)

        self.meas = qms.Measurements (tasks = tasks)

        self.sim_name = 'L{}_dt{}_t_span{}-{}_IC{}'.format ( \
                L, dt, t_span[0], t_span[1], ''.join([ic[0] for ic in IC]))
        print(self.sim_name)
        self.states_path = output_dir + '/states/' + self.sim_name + '_states'
        self.meas_path = output_dir + '/states/' + self.sim_name + '_measures'

        return

    def run_sim (self):
        self.model.time_evolve (self.nmax)
        self.model.write_out ()
        self.meas.take_measurements (self.model.state_list, self.dt)
        self.meas.write_out ()
