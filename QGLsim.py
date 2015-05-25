#!/usr/bin/python

'''
This program time evolves the QGL lattice state and outputs
a single file with all the states in order
'''

import time as lap
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as la
from itertools import permutations
import json
import sys
import errno

from functools import reduce
import QGLsettings as const




def fockinit(dec):
    bin = list('{0:0b}'.format(dec))
    bin = ['0']*(const.L-4-len(bin)) + bin 
    bin = [el.replace('0','dead').replace('1','alive') for el in bin]
    bin = [const.BC[0]]+[const.BC[1]]+bin+[const.BC[2]]+[const.BC[3]]
    init = [const.OPS[key] for key in bin]
    return init

# Kronecker product together list of matricies
def spmatKron(matlist):
    return sp.csc_matrix(reduce(lambda A,B: sp.kron(A,B,'csc'),matlist))

def matKron(matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)

# Generate operators and Hamiltonian
permList2=[]
for perm in permutations(['nbar','nbar','n','n'],4):
    permList2.append(perm)

permList3=[]
for perm in permutations(['nbar','n','n','n'],4):
    permList3.append(perm)
permList2=list(set(permList2))
permList3=list(set(permList3))

def N3(i):
    n3=0
    eyeList=['I']*const.L
    for tup in permList3:
        localOpList3=[tup[0],tup[1],'mix',tup[2],tup[3]]
        prodList3=['I']*(i-2)+localOpList3+(['I']*(const.L-i-3))
        prodList3=[const.OPS[key] for key in prodList3]
        n3=n3+spmatKron(prodList3)
    return n3

def N2(i):
    n2=0
    for tup in permList2:
        localOpList3=[tup[0],tup[1],'mix',tup[2],tup[3]]
        prodList3=['I']*(i-2)+localOpList3+(['I']*(const.L-i-3))
        prodList3=[const.OPS[key] for key in prodList3]
        n2=n2+spmatKron(prodList3)
    return n2

def make_hamiltonian():
    print('making and saving Hamiltonian')
    tic=lap.time()
    ham=0
    for it in range(2,const.L-2):
        ham=ham+N2(it)+N3(it)
        target=open(const.HAM_PATH+'L'+str(const.L)+'_ham.mtx','w')
    #sio.mmwrite(target,ham)
    toc=lap.time()
    print('Hamiltonian took ' + str(toc-tic) + ' s')
    return ham

def make_propagator():
    print('taking matrix exponential...')
    tic=lap.time()
    prop = la.expm(-1j*const.DT*make_hamiltonian())
    toc=lap.time()
    print('propagator took '+str(toc-tic)+' s')
    #with open(const.HAM_PATH+'L'+str(const.L)+'_dt'+str(const.DT_STRING)+'_prop.mtx','w') as outfile:
     #   sio.mmwrite(outfile,prop)
    return prop

def writed(fname,dat):
    with open(const.DATA_PATH+const.SIM_NAME+fname,'wt') as outfile:
        json.dump(dat,outfile)

def readd(fname):
    with open(const.DATA_PATH+const.SIM_NAME+fname,'r') as infile:
        dat = json.load(infile)
    return dat

def make_states(fname):
    slist=np.array(readd(fname))
    return slist[:,0]+1j*slist[:,1]



# simulation loop to return a list of rho's in sparse form
# for wach timestep
def sim():
    dec = const.INIT
    U0 = make_propagator()
    etic=lap.time()
    statelist=[0]*const.NSTEPS
    init = fockinit(dec)
    state=matKron(init)
    for it in range(const.NSTEPS):
        statelist[it]=[state.real,state.imag]
        state=U0.dot(state)
    statelist = np.asarray(statelist)
    etoc=lap.time()
    print('time evolution took '+str(etoc-etic)+' s')
    writed('IC'+str(dec)+'_states.txt',statelist.tolist())
    return statelist[:,0]+1j*statelist[:,1]

