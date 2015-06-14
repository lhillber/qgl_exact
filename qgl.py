#!/usr/bin/python

from multiprocessing import Pipe
from multiprocessing import Process

from itertools import permutations
from functools import reduce
from os.path import isfile
from math import log, sqrt
import time as lap
import os
from collections import OrderedDict
import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse.linalg as spsla
import scipy.fftpack as spf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import epl
import qglopsfuncs as qof
import networkmeasures as nm
import qglio as qio



#==============================================================================
# Model class:
# calculates Hamiltonian and propagator for some L (number of sites) and dt
# (time step)
#==============================================================================
class Model():
    def __init__(self, L, dt, model_dir = '~/Documents/qgl_ediag/models'):
        self.L = L
        self.dt = dt
        self.ham_path  = model_dir + '/hamiltonians'
        self.prop_path = model_dir + '/propogators'
        self.gen_model()
        return

    def gen_model(self):
        self.build_hamiltonian()
        self.build_propagator()

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
            for k in range(self.L):
                H = H+self.N2(k)+self.N3(k)
            sio.mmwrite(self.ham_path,H)
        self.ham = np.array(H)
        return self.ham


    def build_propagator(self):
        """
        Build propagator and export it or import an existing one
        """
        if isfile(self.prop_path):
            print('Importing Propagator...')
            U0 = qio.read_prop(self.prop_path)
        else:
            print('Building propagator...')
            H = self.build_hamiltonian()
            print('computing matrix exponential...')
            tic = lap.time()
            U0 = spsla.expm(-1j*self.dt*H).todense().tolist()
            toc = lap.time()
            print('Matrix exp took '+str(toc-tic)+' s')
            qio.write_prop(self.prop_path,U0)
        self.prop = np.array(U0)
        return self.prop

#==============================================================================
# Measurements class
# Measures applied at each time step
#==============================================================================

class Measurements():
    def __init__(self, tasks=['t','n','MI','nn','BD']):
        self.tasks = tasks
        self.results = {key:[] for key in tasks }
        return

    def __getitem__(self,key):
        return self.results[key]

# reduced density matrix (rdm) for sites in klist
# [1,2] for klist would give rho1_2
    def rdm(self, state, klist):
        """
        Reduced density matrix
        state:
            A full lattice state
        klist:
            a list of sites to keep after tracing out the rest
        """
        L = int(log(len(state),2))
        n = len(klist)
        rest = np.setdiff1d(np.arange(L),klist)
        ordering = []
        ordering = klist+list(rest)
        block = state.reshape(([2]*L))
        block = block.transpose(ordering)
        block = block.reshape(2**n,2**(L-n))
        RDM = np.zeros((2**n,2**n),dtype=complex)
        tot = complex(0,0)
        for i in range(2**n-1):
            Rii = sum(np.multiply(block[i,:],np.conj(block[i,:])))
            tot = tot+Rii
            RDM[i][i] = Rii
            for j in range(i,2**n):
                if i != j:
                    Rij = np.inner(block[i,:],np.conj(block[j,:]))
                    RDM[i][j] = Rij
                    RDM[j][i] = Rij
        RDM[2**n-1,2**n-1] = complex(1,0)-tot

        return RDM


    def ncalc(self, state):
        """
        The set of local number operator measures
        Returns a dictionarry of results with keys
        'nexp' for the expectation value of the number operator at each site
        'DIS' for nexp discritized at .5
        'DIV' for diversity (defined in epl)
        'DEN' for average number density

        state:
            A full lattice state
        """
        L = int(log(len(state),2))
        nexplist = [self.expval(state,self.Ni(k,L)) for k in range(L)]
        print(nexplist)
        dis = [0 if ni<0.5 else 1 for ni in nexplist]
        den = qof.average(nexplist)
        div = epl.diversity(epl.cluster(dis))

        return {'nexp':nexplist,'DIS':dis,'DIV':div,'DEN':den}


    def Ni (self, k, L):
        """
        Represent a local number op in the full Hilbert space
        k:
            Site to construct the number operator for
        L:
            Length of the lattice
        """
        eyelist = np.array(['I']*L)
        eyelist[k] = 'n'
        matlist_N = [qof.ops(key) for key in eyelist]

        return qof.spmatkron(matlist_N)


    def expval (self, state, mat):
        """
        Expectation value of an observable with matrix representation

        state:
            Full state to take expectation value with respect to
        mat:
            Matrix representation of observable in full Hilbert space
        """
        return np.real(qof.dagger(state).dot(mat*state))[0][0]


    def entropy (self, prho):
        """
        Von Neumann entropy of a density matrix
        prho:
            a density matrix
        """
        evals = sla.eigvalsh(prho)
        s = -sum(el*log(el,2) if el > 1e-14 else 0.  for el in evals)

        return s


    def MInetwork(self, state):
        """
        Calculate MI network
        state:
            Full lattice state
        """
        L = int(log(len(state),2))
        MInet = np.zeros((L,L))

        for i in range(L):
            #MI = self.entropy(self.rdm(state,[i]))
            MI = 0.
            MInet[i][i] = MI

            for j in range(i,L):
                if i != j:
                    MI = .5*(self.entropy(self.rdm(state,[i]))+self.entropy(self.rdm(state,[j]))-self.entropy(self.rdm(state,[i,j])))
                if MI > 1e-14:
                    MInet[i][j] = MI
                    MInet[j][i] = MI

        return MInet


    def MIcalc (self, state):
        """
        Create a dictionary of measures with keys
        'net' for MI network
        'CC' for clustering coefficient
        'ND' for network density
        'Y' for disoarity
        'HL' for harmonic lenth
        state:
            Full lattice state
        """

        MInet = self.MInetwork(state)
        MICC = nm.clustering(MInet)
        MIdensity = nm.density(MInet)
        MIdisparity = nm.disparity(MInet)
        MIharmoniclen = nm.harmoniclength(nm.distance(MInet))

        return {'net':MInet.tolist(),'CC':MICC,'ND':MIdensity,'Y':MIdisparity,'HL':MIharmoniclen}


    def nncorrelation (self, state,i,j):
        L = int(log(len(state),2))
        return self.expval(state,self.Ni(i,L).dot(self.Ni(j,L)))-self.expval(state,self.Ni(i,L))*self.expval(state,self.Ni(j,L))


    def nnnetwork (self, state):
        L = int(log(len(state),2))
        nnnet = np.zeros((L,L))

        for i in range(L):
            #nnii = self.nncorrelation(state,i,i)
            nnii = 0
            nnnet[i][i] = nnii

            for j in range(i,L):
                if i != j:
                    nnij = abs(self.nncorrelation(state,i,j))
                    if nnij>1e-14:
                        nnnet[i][j] = nnij
                        nnnet[j][i] = nnij

        return np.fabs(nnnet)


    def nncalc (self, state):
        nnnet = self.nnnetwork(state)
        nnCC = nm.clustering(nnnet)
        nndensity = nm.density(nnnet)
        nndisparity = nm.disparity(nnnet)
        nnharmoniclen = nm.harmoniclength(nm.distance(nnnet))

        return {'net':nnnet.tolist(),'CC':nnCC,'ND':nndensity,'Y':nndisparity,'HL':nnharmoniclen}


    def bdcalc (self, state):
        L = int(log(len(state),2))
        klist = [[i for i in range(mx)] if mx <= round(L/2) else np.setdiff1d(np.arange(L),[i for i in range(mx)]).tolist() for mx in range(1,L)]

        return [np.count_nonzero(filter(lambda el: el > 1e-14, sla.eigvalsh(self.rdm(state,ks)))) for ks in klist ]


    def measure (self, state, t):
        """
        Carry out measurements on state of the system
        """

        for key in self.tasks:

            if key == 'n':
                self.results[key].append(self.ncalc(state))

            elif key == 'MI':
                self.results[key].append(self.MIcalc(state))

            elif key == 't':
                self.results[key].append(t)

            elif key == 'BD':
                self.results[key].append(self.bdcalc(state))

            elif key == 'nn':
                self.results[key].append(self.nncalc(state))

        return


#==============================================================================
# Simulation class
# run time evolution on an initial state and save the result out
#==============================================================================

class Simulation():
    def __init__ (self, tasks, L, tspan, dt, IC, output_dir,
                    model_dir = '~/Documents/qgl_ediag/'):

        os.makedirs(output_dir, exist_ok=True)

        self.tasks = tasks
        self.L = L
        self.dt = dt
        self.nmin = round(tspan[0]/self.dt)
        self.nmax = round(tspan[1]/self.dt)
        self.nsteps = self.nmax - self.nmin

        self.IC = self.make_state(IC)

        self.model = Model (L, dt, model_dir = model_dir)

        self.meas = Measurements (tasks=tasks)

        # states_name = 'L'+str(self.L)+'_dt'+str(self.dt)+'_tspan'+str(tspan[0])+'-'+str(tspan[1])+'_IC'+self.IC_string(IC)+'_states.json'
        # self.states_path = model_dir + "/states/" + states_name
        # ?
        self.states_path = '../data/states/' if states_path is None else states_path
        self.states_name = 
        self.states_path = self.states_path + self.states_name
        self.meas_path = '../data/measures/' if meas_path is None else meas_path
        self.meas_name = 'L'+str(self.L)+'_dt'+str(self.dt)+'_tspan'+str(tspan[0])+'-'+str(tspan[1])+'_IC'+self.IC_string(IC)+'_meas.json'
        self.meas_path = self.meas_path+self.meas_name

        return


    def IC_string (self,IC):
        return '_'.join(['{}_{}'.format(k,v) for k,v in IC.items()])


    def fock (self, dec):
        """
        Generate a Fock state

        dec:
            A base 10 number which identifies the fock state when converted into
            binary (zeros are dead; ones are alive)
        """
        bin = list('{0:0b}'.format(dec))
        bin = ['0']*(self.L-len(bin))+bin
        bin = [el.replace('0','dead').replace('1','alive') for el in bin]
        print(bin)

        return qof.matkron([qof.ops(key) for key in bin])


    def GHZ (self):
        """
        Generate GHZ state of size L
        """
        s1=['alive']*(self.L)
        s2=['dead']*(self.L)
        return (qof.matkron([qof.ops(key) for key in s1])+qof.matkron([qof.ops(key) for key in s2]))*1./sqrt(2.)


    def one_alive (self,k):
        """
        generate a state with one livng site at k
        """

        base = ['dead']*self.L
        base[k] = 'alive'

        return qof.matkron([qof.ops(key) for key in base])

    def W (self):
        """
        Generate W state of size L
        """

        return  1/sqrt(self.L)*sum([self.one_alive(k) for k in range(self.L)])


    def all_alive (self):
        dec = sum(2**n for n in range(0,self.L))

        return self.fock(dec)


    def make_state (self, state):
        lego_states = { 'd': self.foc(int(''.join(ICchars[1:]))), \
                        'G': self.GHZ(), \
                        'W': self.W(), \
                        'a': self.all_alive() }
        return sum ([coeff * lego_states[lego_state] \
                    for (lego_state, coeff) in state])


    def evolve_state (self, start_state=None,):
        """
        Evolving the initial state forward in time and collecting measures of
        the current state along the way

        start_state:
            A full lattice site

        """

        start_state = self.IC if start_state is None else start_state
        if isfile(self.states_path):
            print('Importing states...')
            state_list = qio.read_cdata(self.states_path)

        else:
            print('Time evolving IC...')

            state_list = [0]*self.nsteps
            state = start_state
            i = 0
            for it in range(self.nmax):
                state = self.model.prop.dot(state)

                if it>=self.nmin:
                    state_list[i] = state
                    i = i+1
            qio.write_cdata(self.states_path,state_list)

        if isfile(self.meas_path):
            print('Importing measurements...')
            myMeas = qio.read_data(self.meas_path)
            self.meas.results = myMeas

        else:
            print('Measuring states...')

            for it in range(self.nsteps):
                tic =lap.time()
                t = it*self.dt + self.tmin
                state = state_list[it]
                self.meas.measure(state,t)
                toc = lap.time()
                print('t = ',t,' took ',toc-tic,'s')

            qio.write_data(self.meas_path, self.meas.results)

        return self.model, self.meas




