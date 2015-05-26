#!/usr/bin/python

import sys
sys.path.insert(0,'/usr/lib/mypyscripts')
import time as lap
import json
from itertools import product
from math import sqrt, log
from functools import reduce
import errno

import numpy as np
import scipy.linalg as sla
import scipy.io as sio
import scipy.sparse as sps
import matplotlib.pyplot as plt

import networkmeasures as nm
import epl
import QGLsettings as const
import QGLsim as qsim

def writed(fname,dat):
    with open(const.DATA_PATH+const.SIM_NAME+fname,'wt') as outfile:
        json.dump(dat,outfile)

def readd(fname):
    with open(const.DATA_PATH+const.SIM_NAME+fname,'r') as infile:
        dat = json.load(infile)
    return dat

# make statelist from json exported simdata
def make_states(fname):
    slist=np.array(readd(fname))
    return slist[:,0]+1j*slist[:,1]

# Statistics
def average(thelist):
    return sum(thelist)/len(thelist)

def standard_deviation(thelist):
    av = average(thelist)
    res2 = 0
    for el in thelist:
        res2 = res2+(av-el)*(av-el)
    return sqrt(res2/len(thelist))

# Kronecker product together list of matricies (sps for sparse)
def spmatkron(matlist):
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))

def matkron(matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)

# herm conjugate of a matrix
def dagger(mat):
    return mat.conj().transpose()

# reduced density matrix (rdm) for sites in klist 
# [1,2] for klist would give rho1_2
def rdm(state,klist):
    n = len(klist)
    rest = np.setdiff1d(np.arange(const.L),klist)
    ordering = []
    ordering = klist+list(rest)
    block = state.reshape(([2]*const.L))
    block = block.transpose(ordering)
    block = block.reshape(2**n,2**(const.L-n))
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
    

# global version of number op for site i
def Ni(i):
    eyeList = np.array(['I']*const.L)
    ey(List[i] = 'n'
    prodListN = [const.OPS[key] for key in eyeList]
    return spmatkron(prodListN)

# expectation value of observable
def expval(state,obsmat):
    return np.real(dagger(state).dot(obsmat*state))[0][0]

# calculates measures on single site number ops at time step it
# 'nexp' is the key for the QGL board, 
# 'DIS' for the discritized board
# 'DIV' for diversity 
# 'DEN' for density of life
def ncalc(it,statelist):
    state = statelist[it]
    nexplist = [expval(state,Ni(k)) for k in range(const.L)]
    dis = [round(ni) for ni in nexplist]
    den = average(nexplist)
    div = epl.diversity(epl.cluster(dis))
    return {'nexp':nexplist,'DIS':dis,'DIV':div,'DEN':den}

# simulation time at time step it
def time(it):
    return it*const.DT

# von Neumann entropy of a density matrix prho
def entropy(prho):
    evals = sla.eigvalsh(prho)
    s = -sum(el*log(el,2) if el > 1e-14 else 0.  for el in evals)
    return s

# generates a mutual information network for a given state
def MInetwork(state):
    MInet = np.zeros((const.L,const.L))
    for i in range(const.L):
        #MI = entropy(rdm(state,[i]))
        MI = 0.
        MInet[i][i] = MI
        for j in range(i,const.L):
            if i != j:
                MI = entropy(rdm(state,[i]))+entropy(rdm(state,[j]))-entropy(rdm(state,[i,j]))
            if MI > 1e-14:
                MInet[i][j] = MI
                MInet[j][i] = MI
    return MInet

# Calculates netork and measures of the MI network at times step it
# 'net' is the key for the network
# 'CC' for clustering coefficient
# 'ND' for network density
# 'Y' for disparity
# 'GL' for geodesic length
def MIcalc(it,statelist):
    state = statelist[it]
    MInet = MInetwork(state)
    MICC = nm.clustering(MInet)
    MIdensity = nm.density(MInet)
    MIdisparity = nm.disparity(MInet)
    MIgeodesiclen = nm.geodesic(nm.distance(MInet))
    return {'net':MInet.tolist(),'CC':MICC,'ND':MIdensity,'Y':MIdisparity,'GL':MIgeodesiclen}

# compute the number-number correlator of state for sites i and j
def nncorrelation(state,i,j):
    return expval(state,Ni(i)*Ni(j))-expval(state,Ni(i))*expval(state,Ni(j))

# Generate number-numer network
def nnnetwork(state):
    nnnet = np.zeros((const.L,const.L))
    for i in range(const.L):
        nnii = nncorrelation(state,i,i)
        #nnii = 0
        nnnet[i][i] = nnii
        for j in range(i,const.L):
            if i != j:
                nnij = abs(nncorrelation(state,i,j))
                nnnet[i][j] = nnij
                nnnet[j][i] = nnij
    return nnnet

# Calculate network measures of the number number network
def nncalc(it, statelist):
    state = statelist[it]
    nnnet = nnnetwork(state)
    nnCC = nm.clustering(nnnet)
    nndensity = nm.density(nnnet)
    nndisparity = nm.disparity(nnnet)
    nngeodesiclen = nm.geodesic(nm.distance(nnnet))
    return {'net':nnnet.tolist(),'CC':nnCC,'ND':nndensity,'Y':nndisparity,'GL':nngeodesiclen}

# calculate bond dimension at time step it
def bdcalc(it,statelist):
    state = statelist[it]
    klist = [[i for i in range(mx)] if mx <= round(const.L/2) else np.setdiff1d(np.arange(const.L),[i for i in range(mx)]).tolist() for mx in range(1,const.L)]
    return [np.count_nonzero(filter(lambda el: el > 1e-14,sla.eigvalsh(rdm(state,ks)))) for ks in klist ]

# collect all measures for a single time step it
# in a dictionary of dictionaries
# 't' is the key for the simulation time
# 'n' for the number op measures
# 'nn' for the number-number correlator measures
# 'MI' for the mutual inforomation measures
# 'BD' for the bond dimension
def output(it,statelist):
    print( 't = '+str(time(it))+' tmax = '+str(const.TMAX))
    tic = lap.time()
    outputdict = {'t':time(it),'n':ncalc(it,statelist),'nn':nncalc(it,statelist),'MI':MIcalc(it,statelist),'BD':bdcalc(it,statelist)}
    toc = lap.time()
    print( str(toc-tic)+'s')
    return outputdict

# loop over output over all time steps
# set importQ=True for post processing only (import measurements)
# False to generate and save measurements
def outputs():
    slist = qsim.sim()  
    outputs = [output(i,slist) for i in range(const.NSTEPS)]
    writed('IC'+str(const.INIT)+'_outputs.txt',outputs)
    return outputs


