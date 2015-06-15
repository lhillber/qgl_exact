#!/usr/bin/python

from functools import reduce
from math import log, sqrt
from itertools import permutations

import epl
import qglopsfuncs as qof
import networkmeasures as nm


# Global constants
# ================
# dictionary of local operators, local basis,
# and permutation lists for N@ and N3 ops
OPS = ({
    'I':np.array([[1.,0.],[0.,1.]]),
    'n':np.array([[0.,0.],[0.,1.]]),
    'nbar':np.array([[1.,0.],[0.,0.]]),
    'mix':np.array([[0.,1.],[1.,0.]]),
    'dead':np.array([[1.,0.]]).transpose(),
    'alive':np.array([[0.,1.]]).transpose(),
    'permutations_3':list(set([perm for perm in
        permutations(['nbar','n','n','n'],4)])),
    'permutations_2':list(set([perm for perm in
        permutations(['nbar','nbar','n','n'],4)]))
})


# States
# ======

class State:
    def fock (self, L, dec):
        state = [el.replace('0', 'dead').replace('1', 'alive')
                for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
        return qof.matkron([OPS[key] for key in state])

    def GHZ (self, L):
        s1=['alive']*(L)
        s2=['dead']*(L)
        return (1.0/sqrt(2.0)) \
                * ((qof.matkron([OPS[key] for key in s1]) \
                    + qof.matkron([OPS[key] for key in s2])))

    def one_alive (self, L, k):
        base = ['dead']*L
        base[k] = 'alive'
        return qof.matkron([OPS[key] for key in base])

    def W (self, L):
        return (1.0/sqrt(L)) \
                * sum ([self.one_alive(k) for k in range(L)])

    def all_alive (self, L):
        dec = sum ([2**n for n in range(0,L)])
        return self.fock(dec)

# *TODO* Fix state_chars
    def make_state (self, L, state):
        state_chars = [s[1] for s in state]
        state_map = { 'd': self.foc(L, int(''.join(state_chars[1:]))),
                      'G': self.GHZ(L),
                      'W': self.W(L),
                      'a': self.all_alive(L) }
        return sum ([coeff * state_map[s] for (s, coeff) in state])


# Measurements
# ============

class Measurements():
    def __init__(self, tasks = ['t','n','MI','nn','BD']):
        self.tasks = tasks
        self.measures = {key:[] for key in tasks }
        return

    def __getitem__(self,key):
        return self.measures[key]

    # Take measurements at each timestep
    # ----------------------------------
    def take_measurements (self, states, dt, nstart = 0, nstop = None, ninc = 1):
        nstop = len(states) if nstop == None else nstop
        return [ self.measure(state, i*dt)
                for i,state in enumerate (states[nstart:nstop:ninc]) ]

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
        matlist_N = [OPS[key] for key in eyelist]

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
                self.measures[key].append(self.ncalc(state))

            elif key == 'MI':
                self.measures[key].append(self.MIcalc(state))

            elif key == 't':
                self.measures[key].append(t)

            elif key == 'BD':
                self.measures[key].append(self.bdcalc(state))

            elif key == 'nn':
                self.measures[key].append(self.nncalc(state))

        return


