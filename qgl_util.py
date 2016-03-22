#!/usr/bin/python3
from math import sqrt
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
from multiprocessing import Process, Pipe


# Global constants
# ================
# dictionary of local operators, local basis,
# and permutation lists for N2 and N3 ops
OPS = ({
    'I':np.array([[1.,0.],[0.,1.]]),
    'n':np.array([[0.,0.],[0.,1.]]),
    'nbar':np.array([[1.,0.],[0.,0.]]),
    'mix':np.array([[0.,1.],[1.,0.]]),
    'dead':np.array([[1.,0.]]).transpose(),
    'alive':np.array([[0.,1.]]).transpose(),
    'es'   : np.array([[1./sqrt(2), 1./sqrt(2)]]).transpose(),
    'permutations_3':list(set([perm for perm in
        permutations(['nbar','n','n','n'],4)])),
    'permutations_2':list(set([perm for perm in
        permutations(['nbar','nbar','n','n'],4)]))
})



# Matrix functions
# ================

# Kroeneker product list of matrices
# ----------------------------------
def matkron (matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)

# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron (matlist):
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))

# Hermitian conjugate
# -------------------
def dagger (mat):
    return mat.conj().transpose()



# Initial State Creation
# ======================
   
# Create Fock state
# -----------------
def fock (L, config, zero = 'dead', one = 'alive'):
    dec = int(config)
    state = [el.replace('0', zero).replace('1', one)
            for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
    return matkron([OPS[key] for key in state])

# Create state with config - > binary: 0 - >dead, 1 -> 1/sqrt(2) (|0> +|1>)
# ------------------------------------------------------------------------
def local_superposition (L, config):
    return fock(L, config, one = 'es')

# Create state with one or two live sites
# ---------------------------------------
def one_alive (L, config):
    dec = 2**int(config)
    return fock(L, dec)

def two_alive(L, config):
    i, j = map(int, config.split('_'))
    return fock(L, 2**i + 2**j)

def two_es(L, config):
    i, j = map(int, config.split('_'))
    return local_superposition(L, 2**i + 2**j)

# Create state with all sites living
# ----------------------------------
def all_alive (L, config):
    dec = sum ([2**n for n in range(0,L)])
    return fock(L, dec)

# Create GHZ state
# ----------------
def GHZ (L, congif):
    s1=['alive']*(L)
    s2=['dead']*(L)
    return (1.0/sqrt(2.0)) \
            * ((matkron([OPS[key] for key in s1]) \
                + matkron([OPS[key] for key in s2])))

# Create W state
# --------------
def W (L, config):
    return (1.0/sqrt(L)) \
            * sum ([one_alive(L, k) for k in range(L)])

# Create as state with sites i and j maximally entangled
# reduces to 1/sqrt(2) (|00> + |11>) in L = 2 limit
# ------------------------------------------------------
def entangled_pair (L, config):
    i, j = map(int, config.split('_'))
    return 1./sqrt(2) * (fock(L, 0) + fock(L, 2**i + 2**j))

def center(L, config):
    len_cent = int(config[0])
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    cent_IC = [(config[1:], 1)]
    left = fock(len_L, 0)
    cent = make_state(len_cent, cent_IC)
    right = fock(len_R, 0)
    if len_back == 0:
        return cent
    elif len_back == 1:
        return matkron([cent, right])
    else:
        return matkron([left, cent, right])

# Make the specified state
# ------------------------

smap = { 'd' : fock,
         'l' : local_superposition,
         't' : two_es,
         'a' : all_alive,
         'c' : center,
         'G' : GHZ,
         'W' : W,
         'E' : entangled_pair } 

def make_state (L, IC):
    state = np.asarray([[0.]*(2**L)]).transpose()
    for s in IC: 
            name = s[0][0]
            config = s[0][1:]
            coeff = s[1]
            state = state + coeff * smap[name](L, config)
    return state



