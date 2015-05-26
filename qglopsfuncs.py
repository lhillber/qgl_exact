#!/usr/bin/python
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
# simulation parameters


def matkron(matlist):
    """
    Kronecker product together a list of matrices and return the resulting
    (dense) matrix
    """
    return reduce(lambda A,B: np.kron(A,B),matlist)

def spmatkron(matlist):
    """
    Kronecker product together a list of matrices and return the resulting
    matrix in csc format
    """
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))



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

def ops(key):
    return OPS[key]


'''

    global L,INIT,BC,TMAX,DT,DT_STRING,NSTEPS,SIM_NAME,DATA_PATH,HAM_PATH,OPS
    L = 7
    BC = ['dead','dead','dead','dead']
    INIT = 3
    TMAX = 5
    DT = 1.
    DT_STRING = '1'
    NSTEPS = int(round(TMAX/DT)+1)
    SIM_NAME = 'L'+str(L)+'_tmax'+str(TMAX)+'_dt'+DT_STRING+'_'
    DATA_PATH = '../data/'
    HAM_PATH = '../data/Hamiltonians/'
    OPS = {'I':np.array([[1.,0.],[0.,1.]]),'n':np.array([[0.,0.],[0.,1.]]),'nbar':np.array([[1.,0.],[0.,0.]]),'mix':np.array([[0.,1.],[1.,0.]]),'dead':np.array([[1.,0.]]).transpose(),'alive':np.array([[0.,1.]]).transpose()}
'''

