#!/usr/bin/python3
from functools import reduce
from math import log, sqrt
from itertools import permutations
import numpy as np
import numpy.linalg as nla
import json
import scipy.linalg as sla
import epl
from qgl_util import *
import networkmeasures as nm


# ==========================================================
# Measurements class
# suite of measurements available to make at each times step
# ==========================================================

class Measurements():
    def __init__(self, tasks, meas_file):
        self.tasks = tasks
        self.measures = {key:[] for key in tasks }
        self.meas_file = meas_file
        return
    # Easy retreval of data
    # ----------------------
    def __getitem__(self,key):
        return self.measures[key]

    # Take measurements at each timestep
    # ----------------------------------
    def take_measurements (self, states, dt, nmin, nmax, ninc = 1):
        return [ self.measure(state, i*dt + nmin*dt) \
                for i,state in enumerate (states[nmin:nmax:ninc]) ]

    # Take measurements at time t
    # ---------------------------
    def measure (self, state, t):
        """
        Carry out measurements on state of the system
        """
        print(t)
        for key in self.tasks:
            if key == 'n':
                self.measures[key].append(self.ncalc(state))

            elif key == 'MI':
                self.measures[key].append(self.MIcalc(state))

            elif key == 't':
                self.measures[key].append(t)

            elif key == 'EC':
                self.measures[key].append(self.entropy_of_cut(state))

            elif key == 'nn':
                self.measures[key].append(self.nncalc(state))
        return
   
    # save measurement results
    # ------------------------
    def write_out(self):
        data = np.asarray(self.measures).tolist()
        with open(self.meas_file, 'w') as outfile:
            json.dump(data, outfile)
        return
    # load measurement results
    # ------------------------
    def read_in(self, meas_file):
        with open(meas_file, 'r') as infile:
           data =  json.load(infile)
        return data
    
    # reduced density matrix (rdm) of sites in klist
    # ex) [1,2] for klist would give rho1_2

    def rdm(self, state, js, ds=None):
        js = np.array(js)
        if ds is None:
            L = int( log(len(state), 2) )
            ds = [2]*L
        else:
            L = len(ds)

        rest = np.setdiff1d(np.arange(L), js)
        ordering = np.concatenate((js, rest))
        dL = np.prod(ds)
        djs = np.prod(np.array(ds).take(js))
        drest = np.prod(np.array(ds).take(rest))

        block = state.reshape(ds).transpose(ordering).reshape(djs, drest)

        RDM = np.zeros((djs, djs), dtype=complex)
        tot = complex(0,0)
        for i in range(djs):
            for j in range(djs):
                Rij = np.inner(block[i,:], np.conj(block[j,:]))
                RDM[i, j] = Rij
        return RDM


    def ncalc(self, state):
        """
        The set of local number operator subasks
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
        dis = [0 if ni<0.5 else 1 for ni in nexplist]
        den = np.mean(nexplist)
        div = epl.diversity(epl.cluster(dis))

        return {'nexp':nexplist,'DIS':dis,'DIV':div,'DEN':den}


    def Ni (self, k, L):
        eyelist = np.array(['I']*L)
        eyelist[k] = 'n'
        matlist_N = [OPS[key] for key in eyelist]

        return spmatkron(matlist_N)


    def expval (self, state, mat):
        """
        Expectation value of an observable with matrix representation

        state:
            Full state to take expectation value with respect to
        mat:
            Matrix representation of observable in full Hilbert space
        """
        return np.real(dagger(state).dot(mat*state))[0][0]


    def entropy (self, prho):
        """
        Von Neumann entropy of a density matrix
        prho:
            a density matrix
        """
        #evals = sla.eigvalsh(prho)
        evals = nla.eigvalsh(prho)
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
                    if MI<= 1e-14:
                        MInet[i][j] = 1e-14
                        MInet[j][i] = 1e-14
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

        MInet            = self.MInetwork(state)
        MICC             = nm.clustering(MInet)
        MIdensity        = nm.density(MInet)
        MIdisparity      = nm.disparity(MInet)
        MIharmoniclen    = nm.harmoniclength(nm.distance(MInet))
        MIeveccentrality = nm.eigenvectorcentralitynx0(MInet)
        return {'net' : MInet.tolist(),
                'CC'  : MICC,
                'ND'  : MIdensity,
                'Y'   : MIdisparity,
                'HL'  : MIharmoniclen,
                'EV'  : list(MIeveccentrality.values())}


    def nncorrelation (self, state,i,j):
        L = int(log(len(state),2))
        return self.expval(state,self.Ni(i,L).dot(self.Ni(j,L)))


    def nnnetwork (self, state):
        L = int(log(len(state),2))
        nnnet = np.zeros((L,L))

        for i in range(L):
            #nnii = self.nncorrelation(state,i,i)
            nnii = 0
            nnnet[i][i] = nnii

            for j in range(i,L):
                if i != j:
                    nnij = self.nncorrelation(state,i,j)
                    if nnij>1e-14:
                        nnnet[i][j] = nnij
                        nnnet[j][i] = nnij
        return nnnet


    def nncalc (self, state):
        nnnet = self.nnnetwork(state)
        nnCC = nm.clustering(nnnet)
        nndensity = nm.density(nnnet)
        nndisparity = nm.disparity(nnnet)
        #nnharmoniclen = nm.harmoniclength(nm.distance(nnnet))

        return {'net':nnnet.tolist(),'CC':nnCC,'ND':nndensity,'Y':nndisparity,'HL':1}


    def entropy_of_cut (self, state):
        L = int(log(len(state),2))
        klist = [ [i for i in range(mx)] if mx <= round(L/2) 
                else np.setdiff1d(np.arange(L), [i for i in range(mx)]).tolist() 
                for mx in range(1,L)]
        return [self.entropy(self.rdm(state, ks)) for ks in klist ]




