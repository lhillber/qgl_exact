#!/usr/bin/python

from itertools import permutations
from functools import reduce
from os.path import isfile
from math import log

import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
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
    def __init__(self,L,TMAX,DT, ham_path=None,prop_path=None):
        self.L = L
        self.TMAX = TMAX
        self.DT = DT
        self.NSTEPS = int(round(self.TMAX/self.DT))
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
    def __init__(self,L ,TMAX, DT, IC, states_path=None, meas_path=None, ham_path=None, prop_path=None):
        """
        model:
            an instance of the Model class (which specifies L and DT)
        meas:
            an instance of the Measurements class
        IC:
            a string describing the initial condition (i.e 'd3','e1:2_4:6bd',
            'G', 'W', 'C', 'theta_GW45',thetaphi45_90)
        """
        self.model = Model(L, TMAX,DT, ham_path=ham_path, prop_path=prop_path)
        self.meas = Measurements()
        self.L = L
        self.TMAX = TMAX
        self.DT = DT
        self.U0 = self.model.prop
        self.IC = IC
        self.myIC = self.make_IC()
       
        self.states_path = \
            '../data/states/''L'+str(self.L)+'_dt'+str(self.DT)+'_tmax'+str(self.TMAX)+'_IC'+self.IC+'_states.json' \
            if states_path is None else states_path
        self.meas_path = \
            '../data/measures/''L'+str(self.L)+'_dt'+str(self.DT)+'_tmax'+str(self.TMAX)+'_IC'+self.IC+'_meas.json' \
            if meas_path is None else meas_path
        self.NSTEPS = self.model.NSTEPS
        return

    def fockinit(self,dec):
        """
        Generate a Fock state

        dec:
            A base 10 number which identifies the fock state when converted into
            binary (zeros are dead; ones are alive)
        """
        bin = list('{0:0b}'.format(dec))
        bin = ['0']*(self.L-len(bin))+bin
        bin = [el.replace('0','dead').replace('1','alive') for el in bin]
        return qof.matkron([qof.ops(key) for key in bin])

    def make_IC(self):
        """
        Parse the IC string to determine which state generator to use
        """
        ICchars = list(self.IC)
        if ICchars[0]=='d':
            dec = int(''.join(ICchars[1:]))
            return self.fockinit(dec)
        if ICchars[0]=='e':
            print('entangled IC not yet implemented')


    def evolve_state(self,start_state=None,mode='use_IC'):
        start_state = self.myIC if start_state is None else start_state
        """
        Evolving the initial state forward in time and collecting measures of
        the current state along the way

        start_state:
            A full lattice site

        mode:
            'TODO'
        """
        if mode=='use_IC':
            if isfile(self.states_path):
                print('Importing states...')
                state_list = qio.read_cdata(self.states_path)


            else:
                print('Time evolving IC...')

                state_list = [0]*self.NSTEPS
                state = start_state
                for it in range(self.NSTEPS):
                    t = it*self.DT
                    state_list[it] = state
                    state = self.U0.dot(state)
                qio.write_cdata(self.states_path,state_list)
            
            if isfile(self.meas_path):
                print('Importing measurements...')
                myMeas = qio.read_data(self.meas_path)
                self.meas.results = myMeas
            else:
                for it in range(self.NSTEPS):
                    t = it*self.DT
                    state = state_list[it]
                    self.meas.measure(state,t)
                qio.write_data(self.meas_path, self.meas.results)
            return self.model, self.meas
        if mode=='load_state':
                print('loading states not yet supported')

#==============================================================================
# Measurements class
# Measures applied at each time step
#==============================================================================

class Measurements():
    def __init__(self, tasks=['t','n','MI']):
        self.tasks = tasks
        self.results = {key:[] for key in tasks }
        return

    def __getitem__(self,key):
        return self.results[key]

# reduced density matrix (rdm) for sites in klist 
# [1,2] for klist would give rho1_2
    def rdm(self,state,klist):
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

    def ncalc(self,state):
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
        dis = [round(ni) for ni in nexplist]
        den = qof.average(nexplist)
        div = epl.diversity(epl.cluster(dis))
        return {'nexp':nexplist,'DIS':dis,'DIV':div,'DEN':den}

    def Ni(self,k,L):
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

    def expval(self,state,mat):
        """
        Expectation value of an observable with matrix representation
        
        state:
            Full state to take expectation value with respect to 

        mat:
            Matrix representation of observable in full Hilbert space
        """
        return np.real(qof.dagger(state).dot(mat*state))[0][0]

    def entropy(self,prho):
        """
        Von Neumann entropy of a density matrix
        prho:
            a density matrix
        """
        evals = sla.eigvalsh(prho)
        s = -sum(el*log(el,2) if el > 1e-14 else 0.  for el in evals)
        return s 

    def MInetwork(self,state):
        """
        Calculate MI network
        state:
            Full lattice state
        """
        L = int(log(len(state),2))
        MInet = np.zeros((L,L))
        for i in range(L):
            MI = self.entropy(self.rdm(state,[i]))
            MI = 0.
            MInet[i][i] = MI
            for j in range(i,L):
                if i != j:
                    MI = self.entropy(self.rdm(state,[i]))+self.entropy(self.rdm(state,[j]))-self.entropy(self.rdm(state,[i,j]))
                if MI > 1e-14:
                    MInet[i][j] = MI
                    MInet[j][i] = MI
        return MInet

    def MIcalc(self,state):
        """
        Create a dictionary of measures with keys
        'net' for MI network
        'CC' for clustering coefficient
        'ND' for network density
        'Y' for disoarity
        'GL' for geodesic path length

        state:
            Full lattice state

        """
        L = int(log(len(state),2))
        MInet =self.MInetwork(state)
        MICC = nm.clustering(MInet)
        MIdensity = nm.density(MInet)
        MIdisparity = nm.disparity(MInet)
        MIgeodesiclen = nm.geodesic(nm.distance(MInet),3,5)
        return {'net':MInet.tolist(),'CC':MICC,'ND':MIdensity,'Y':MIdisparity,'GL':MIgeodesiclen}



    def measure(self,state,t):
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
        return


#==============================================================================
# Plotting class takes a full measurement class
#==============================================================================

class Plotting():
    def __init__(self,model,meas):
        self.results = meas.results
        self.times = meas.results['t'] 
        self.L = model.L
        self.TMAX = model.TMAX
        self.DT = model.DT
        self.NSTEPS = model.NSTEPS

        self.xtick_locs = range(0,self.NSTEPS,int(self.NSTEPS/10))
        self.xtick_lbls = [self.times[i] for i in self.xtick_locs]
        self.ytick_locs = range(0,self.L,int(self.L/5))
        self.ytick_lbls = [loc+1 for loc in self.ytick_locs]

    def n_results(self):
        n_res = self.results['n']
        nexp_board = np.array([n_res[i]['nexp'] for i in range(self.NSTEPS)]).transpose()
        den_res = np.array([n_res[i]['DEN'] for i in range(self.NSTEPS)])
        dis_board = np.array([n_res[i]['DIS'] for i in range(self.NSTEPS)]).transpose()
        div_res = np.array([n_res[i]['DIV'] for i in range(self.NSTEPS)])
        return nexp_board, dis_board, den_res, div_res

    def n_plots(self):
        n_res = self.n_results()
        
        fig1 = plt.figure(1)
        plt.subplot(211)
        plt.imshow(n_res[0],cmap=plt.cm.gray_r,interpolation='none')
        plt.xlabel('Time')
        plt.ylabel('Lattice Site')
        plt.title('The Quantum Game of Life')
        plt.xticks(self.xtick_locs,self.xtick_lbls)
        plt.yticks(self.ytick_locs,self.ytick_lbls)
        plt.subplot(212)
        plt.imshow(n_res[1],cmap=plt.cm.gray_r,interpolation='none',)
        plt.xlabel('Time')
        plt.ylabel('Lattice Site')
        plt.title('Discretized QGL')
        plt.xticks(self.xtick_locs,self.xtick_lbls)
        plt.yticks(self.ytick_locs,self.ytick_lbls)
        plt.tight_layout()

        fig2 = plt.figure(2)
        plt.subplot(121)
        plt.plot(self.times,n_res[2])
        plt.title('Average population density')
        plt.subplot(122)
        plt.plot(self.times,n_res[3])
        plt.title('Diversity')
        return fig1, fig2


    def MI_results(self):
        MI_res = self.results['MI']
        MICC_res = np.array([MI_res[i]['CC'] for i in range(self.NSTEPS)])
        MIND_res = np.array([MI_res[i]['ND'] for i in range(self.NSTEPS)])
        MIY_res = np.array([MI_res[i]['Y'] for i in range(self.NSTEPS)])
        MIGL_res = np.array([MI_res[i]['GL'] for i in range(self.NSTEPS)])
        return MICC_res, MIND_res, MIY_res, MIGL_res

    def MI_plots(self):
        MI_res = self.MI_results()

        fig3 = plt.figure(3)
        plt.subplot(221)
        plt.plot(self.times, MI_res[0])
        plt.title('Clustering Coefficient')
        plt.subplot(222)
        plt.plot(self.times, MI_res[1])
        plt.title('Network Density')
        plt.subplot(223)
        plt.plot(self.times, MI_res[2])
        plt.title('Disparity')
        plt.subplot(224)
        plt.plot(self.times, MI_res[3])
        plt.title('Geodesic Length (3-5)')
        plt.ylim([0.,30.])
        return fig3

mySim = Simulation(8,10,.1,'d13')
myMod,myMeas = mySim.evolve_state()

myPlots = Plotting(myMod,myMeas)

myPlots.n_plots()
myPlots.MI_plots()
plt.show()


