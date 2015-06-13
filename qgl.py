#!/usr/bin/python

import multiprocessing as mp

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
    def __init__(self, L, TSPAN, DT, ham_path=None, prop_path=None):
        self.L = L
        self.TMAX = TSPAN[1]
        self.DT = DT
        self.ham_path ='../data/hamiltonians/L'+str(self.L)+'_ham.mtx'  \
                if ham_path is  None else ham_path

        self.prop_path = '../data/propagators/L'+str(self.L)+'_dt'+str(self.DT)+'_prop.txt' \
                if prop_path is None else prop_path
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
            U0 = qio.read_cdata(self.prop_path)
        else:
            print('Building propagator...')
            H = self.build_hamiltonian()
            U0 = spsla.expm(-1j*self.DT*H).todense().tolist()
            qio.write_cdata(self.prop_path,U0)
        self.prop = np.array(U0)
        return self.prop

#==============================================================================
# Simulation class
# run time evolution on an initial state and save the result out
#==============================================================================

class Simulation():
    def __init__(self,tasks, L ,TSPAN, DT, IC, states_path=None, meas_path=None, ham_path=None, prop_path=None):
        """
        model:
            an instance of the Model class (which specifies L and DT)
        meas:
            an instance of the Measurements class
        IC:
            a string describing the initial condition (i.e 'd3','e1:2_4:6bd',
            'G', 'W', 'C', 'theta_GW45',thetaphi45_90)
        """
        self.L_list = L_list
        self.new_sim(0)

    def new_sim(n):
        self.model = Model(L,TSPAN, DT, ham_path=ham_path, prop_path=prop_path)
        self.meas = Measurements(tasks=tasks)
        self.L = self.L_list(n)
        self.TMAX = selt.TMAX_list[n]
        TSPAN[1]
        self.TMIN = TSPAN[0]
        self.TSPAN = TSPAN
        self.DT = DT
        self.NMIN = round(self.TMIN/self.DT)
        self.NMAX = round(self.TMAX/self.DT)
        self.NSTEPS = self.NMAX - self.NMIN
        self.U0 = self.model.prop
        self.IC = IC
        self.myIC = self.make_IC()
        self.model.IC = IC
        self.tasks = tasks

        self.states_path = '../data/states/' if states_path is None else states_path
        self.states_name = 'L'+str(self.L)+'_dt'+str(self.DT)+'_tspan'+str(self.TSPAN[0])+'-'+str(self.TSPAN[1])+'_IC'+self.IC_string(IC)+'_states.json'
        self.states_path = self.states_path + self.states_name
        self.meas_path = '../data/measures/' if meas_path is None else meas_path
        self.meas_name = 'L'+str(self.L)+'_dt'+str(self.DT)+'_tspan'+str(self.TSPAN[0])+'-'+str(self.TSPAN[1])+'_IC'+self.IC_string(IC)+'_meas.json'
        self.meas_path = self.meas_path+self.meas_name
        return

    def IC_string(self,IC):
        return '_'.join(['{}_{}'.format(k,v) for k,v in IC.items()])


    def fock(self, dec):
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

    def GHZ(self):
        """
        Generate GHZ state of size L
        """
        s1=['alive']*(self.L)
        s2=['dead']*(self.L)
        return (qof.matkron([qof.ops(key) for key in s1])+qof.matkron([qof.ops(key) for key in s2]))*1./sqrt(2.)

    def one_alive(self,k):
        """
        generate a state with one livng site at k
        """

        base = ['dead']*self.L
        base[k] = 'alive'
        return qof.matkron([qof.ops(key) for key in base])

    def W(self):
        """
        Generate W state of size L
        """
        return  1/sqrt(self.L)*sum([self.one_alive(k) for k in range(self.L)])

    def all_alive(self):
        dec = sum(2**n for n in range(0,self.L))
        return self.fock(dec)

    def make_IC(self):
        """
        Parse the IC string to determine which state generator to use
        """
        states = self.IC.keys()
        mystate = 0
        for state in states:
            ICchars = list(state)
            if ICchars[0]=='d':
                dec = int(''.join(ICchars[1:]))
                mystate = mystate + self.IC[state]*self.fock(dec)
            elif ICchars[0]=='G':
                mystate = mystate + self.IC[state]*self.GHZ()
            elif ICchars[0]=='W':
                mystate = mystate + self.IC[state]*self.W()
            elif ICchars[0]=='a':
                mystate = mystate + self.IC[state]*self.all_alive()
        return mystate

    def evolve_state(self, start_state=None, mode='use_IC'):
        """
        Evolving the initial state forward in time and collecting measures of
        the current state along the way

        start_state:
            A full lattice site

        mode:
            'TODO'
        """

        start_state = self.myIC if start_state is None else start_state
        if mode=='use_IC':
            if isfile(self.states_path):
                print('Importing states...')
                state_list = qio.read_cdata(self.states_path)

            else:
                print('Time evolving IC...')
                state_list = [0]*self.NSTEPS
                state = start_state
                i = 0
                for it in range(self.NMAX):

                    state = self.U0.dot(state)
                    if it>=self.NMIN:
                        state_list[i] = state
                        i = i+1
                qio.write_cdata(self.states_path,state_list)

            if isfile(self.meas_path):
                print('Importing measurements...')
                myMeas = qio.read_data(self.meas_path)
                self.meas.results = myMeas
            else:
                print('Measuring states...')
                for it in range(self.NSTEPS):
                    tic =lap.time()
                    t = it*self.DT + self.TMIN
                    state = state_list[it]
                    self.meas.measure(state,t)
                    toc = lap.time()
                    print('t = ',t,'took',toc-tic,'s')

                qio.write_data(self.meas_path, self.meas.results)
            return self.model, self.meas
        if mode=='load_state':
                print('loading states not yet supported')

    def run_sims():
        self.setup(1)
        selt.evolve_state()



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

    def Ni(self, k, L):
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

    def expval(self, state, mat):
        """
        Expectation value of an observable with matrix representation

        state:
            Full state to take expectation value with respect to
        mat:
            Matrix representation of observable in full Hilbert space
        """
        return np.real(qof.dagger(state).dot(mat*state))[0][0]

    def entropy(self, prho):
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

    def MIcalc(self, state):
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

        L = int(log(len(state),2))
        MInet =self.MInetwork(state)
        MICC = nm.clustering(MInet)
        MIdensity = nm.density(MInet)
        MIdisparity = nm.disparity(MInet)
        MIharmoniclen = nm.harmoniclength(nm.distance(MInet))
        return {'net':MInet.tolist(),'CC':MICC,'ND':MIdensity,'Y':MIdisparity,'HL':MIharmoniclen}

    def nncorrelation(self, state,i,j):
        L = int(log(len(state),2))
        return self.expval(state,self.Ni(i,L).dot(self.Ni(j,L)))-self.expval(state,self.Ni(i,L))*self.expval(state,self.Ni(j,L))

    def nnnetwork(self, state):
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

    def nncalc(self, state):
        nnnet = self.nnnetwork(state)
        nnCC = nm.clustering(nnnet)
        nndensity = nm.density(nnnet)
        nndisparity = nm.disparity(nnnet)
        nnharmoniclen = nm.harmoniclength(nm.distance(nnnet))
        return {'net':nnnet.tolist(),'CC':nnCC,'ND':nndensity,'Y':nndisparity,'HL':nnharmoniclen}

    def bdcalc(self, state):
        L = int(log(len(state),2))
        klist = [[i for i in range(mx)] if mx <= round(L/2) else np.setdiff1d(np.arange(L),[i for i in range(mx)]).tolist() for mx in range(1,L)]
        return [np.count_nonzero(filter(lambda el: el > 1e-14, sla.eigvalsh(self.rdm(state,ks)))) for ks in klist ]


    def measure(self, state, t):
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

def run_sim (L_list = [10], dt_list = [1.0], tasks = ['t', 'n', 'nn', 'MI'],
        output_dir = "output", t_span_list = [(0.0, 10.0), (2.0, 12.0)],
        IC_list = [[('a',1.0),('W',0.0)]]):

    PLOTS_PATH = output_dir + '/plots/'
    DATA_PATH  = output_dir + '/data/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH,  exist_ok=True)

    mySim = []
    for L in L_list:
        for dt in dt_list:
            for t_span in t_span_list:
                for IC in IC_list:
                    IC = OrderedDict(IC)
                    mySim.append(Simulation(tasks, L, tspan, dt, IC,
                            states_path = DATA_PATH, meas_path = DATA_PATH))

    p = mp.Pool(len(mySims))
    p.map(Simulation.evolve_state, mySims)
