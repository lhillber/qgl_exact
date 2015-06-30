#!/usr/bin/python
from math import sqrt
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
from multiprocessing import Process, Pipe

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



def matkron (matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)



# Initial State Creation
# ======================

# Create Fock state
# -----------------
def fock (L, config, zero = 'dead', one = 'alive':
    dec = int(config)
    state = [el.replace('0', zerro).replace('1', one)
            for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
    return matkron([OPS[key] for key in state])

# Create state with config - > binary: 0 - >dead, 1 -> 1/sqrt(2) (|0> +|1>)
# ------------------------------------------------------------------------
def local_superposition (L, config):
    return fock(L, config, one = 'es')

# Create state with one or two live sites
# ---------------------------------------
def one_alive (L, k):
    return fock(L, 2**k)

def two_alive(L, i, j):
    return fock(L, 2**i + 2**j)

def two_es (L, i, j)
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

# Create as state with a singlet between sites i and j
# ----------------------------------------------------
def entangled_pair (L, config):
    i, j = map(int, config.split('_'))
    return 1./sqrt(2) * (fock(L, 0) + two_alive(L, i , j))

# Make the specified state
# ------------------------
def make_state (L, IC):
    smap = { 'd': fock,
             'a': all_alive, 
             'G': GHZ,
             'W': W,
             'E': entangled_pair } 
    
    state = np.asarray([[0.]*(2**L)]).transpose()
    for s in IC: 
            name = s[0][0]
            config = s[0][1:]
            coeff = s[1]
            state = state + coeff * smap[name](L, config)
    return state




print(make_state(2, [('E0_1',1.0)]))
