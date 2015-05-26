#!/usr/bin/python

from itertools import permutations
from functools import reduce
from os.path import isfile

import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.sparse.linalg as spsla

import qglopsfuncs as qof
import qglio as qio
#==============================================================================
# Model class:
# calculates Hamiltonian and propagator for some L (number of sites) and dt
# (time step)
#==============================================================================
class Model():
    def __init__(self,L,DT):
        self.L = L
        self.DT = DT
        self.ham = []
        self.prop = []
        self.ham_path ='../data/hamiltonians/L'+'_ham.mtx'

        self.prop_path = '../data/propagators/L'+str(self.L)+'_dt'+str(self.DT)+'_prop.txt'
        return


    def add_ham(self,mat):
        """
        mat:
            matrix representation of Hamiltonian
        """
        self.ham.append(mat)
        return

    
    def add_prop(self,mat):
        """
        mat:
            matrix representation of propagator
        """
        self.prop.append(mat)
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
            matlist3 = ['I']*(k-2)+local_matlist3+(['I']*(self.L-k-3))
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
            matlist2 = ['I']*(k-2)+local_matlist2+(['I']*(self.L-k-3))
            matlist2 = [qof.ops(key) for key in matlist2]
            n2 = n2 + qof.spmatkron(matlist2)
        return n2

    
    def build_hamiltonian(self):
        """
        Build Hamiltonian and export it or import an existing one
        """
        
        if isfile(self.ham_path):
            print('Imporiting Hamiltonian...')
            H = sio.mmread(self.ham_path).tocsc()
        else:
            print('Building Hamiltonian...')
            H = 0
            for k in range(2,self.L-2):
                H = H+self.N2(k)+self.N3(k)
            print('Saving Hamiltonian...')
            sio.mmwrite(self.ham_path,H)
        return H
    
    
    def build_propagator(self):
        """
        Build propagator and export it or import an existing one
        """
        if isfile(self.prop_path):
            print('Importing Propagator...')
            U0 = qio.read_cdata(self.prop_path)
        else:
            print('building propagator...')
            H = self.build_hamiltonian()
            U0 = spsla.expm(-1j*self.DT*H).todense().tolist()
            print('saving propagator...')
            qio.write_cdata(self.prop_path,U0)
        return np.array(U0)

#==============================================================================
# Simulation class
# run time evolution on an initial state and save the result out
#==============================================================================

class Simulation():
    def __init__(self,model,TMAX,IC):
        """
        model:
            an instance of the Model class (which specifies L and DT)
        TMAX:
            a number specifying the max simulation time
        IC:
            a string describing the initial condition (i.e 'd3','i1:2_4:6',
            'G', 'W', 'C', 'theta_GW45',thetaphi45_90)
        """
        self.model=model
        self.U0 = model.prop[0]
        self.L = self.model.L
        self.DT = self.model.DT
        self.TMAX = TMAX
        self.IC = IC
        self.states_path = '../data/states/''L'+str(self.L)+'_dt'+str(self.DT)+'_tmax'+str(self.TMAX)+'_IC'+self.IC+'_states.json'
        self.NSTEPS = int(round(self.TMAX/self.DT)+1)
        return

    def fockinit(self,dec):
        bin = list('{0:0b}'.format(dec))
        bin = ['0']*(self.L-1-len(bin))+bin
        bin = [el.replace('0','dead').replace('1','alive') for el in bin]
        return qof.matkron([qof.ops(key) for key in bin])

    
    def make_IC(self):
        ICchars = list(self.IC)
        if ICchars[0]=='d':
            dec = int(''.join(ICchars[1:]))
            return self.fockinit(dec)
        if ICchars[0]=='i':
            print('entangled IC not yet implemented')
    
    
    def evolve_state(self,start_state):
        if isfile(self.states_path):
            print('Importing states...')
            states = qio.read_cdata(self.states_path)
        state_list = [0]*self.NSTEPS
        state = start_state
        for it in range(self.NSTEPS):
            state_list[it] = state
            state = self.U0.dot(state)
        qio.write_cdata(self.states_path,state_list)
        return state_list





m = Model(10,.1)
U0 = m.build_propagator()
m.add_prop(U0)

s=Simulation(m,10,'d11')
s.evolve_state(s.make_IC())
