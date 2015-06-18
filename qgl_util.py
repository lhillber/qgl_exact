#!/usr/bin/python
from math import sqrt
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations



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

# List average (should replace with np.mean)
# ------------------------------------------
def average (mylist):
    return sum(mylist)/len(mylist)



# Initial State Creation
# ======================

# Create Fock state
# -----------------
def fock (L, dec):
    state = [el.replace('0', 'dead').replace('1', 'alive')
            for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
    return matkron([OPS[key] for key in state])

# Create GHZ state
# ----------------
def GHZ (L):
    s1=['alive']*(L)
    s2=['dead']*(L)
    return (1.0/sqrt(2.0)) \
            * ((matkron([OPS[key] for key in s1]) \
                + matkron([OPS[key] for key in s2])))

# Create state with single live site
# ----------------------------------
def one_alive (L, k):
    base = ['dead']*L
    base[k] = 'alive'
    return matkron([OPS[key] for key in base])

# Create W state
# --------------
def W (L):
    return (1.0/sqrt(L)) \
            * sum ([one_alive(L, k) for k in range(L)])

# Create state with all sites living
# ----------------------------------
def all_alive (L):
    dec = sum ([2**n for n in range(0,L)])
    return fock(L, dec)

# Dict of available states
# ------------------------
def state_map (key, L, dec):
    smap = { 'd': fock(L, dec),
             'G': GHZ(L),
             'W': W(L),
             'a': all_alive(L) }
    return smap[key]

# Make the specified state
# ------------------------
def make_state (L, state):
    state_chars = [s[0] for s in state]
    state_chars_list = map(lambda x: list(x), state_chars)
    state_coeffs = [s[1] for s in state] 
    ziped = zip(state_chars_list, state_coeffs)
    
    state = np.asarray([[0]*(2**L)]).transpose()
    for (s, coeff) in ziped:
        if len(s)>1:
            dec = int(''.join(s[1:]))
        else: 
            dec = 0

        state = state + coeff * state_map(s[0], L, dec)
    return state



# Parallelize
# ===========
# http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class

# Wrapper for multiprocessing
# ---------------------------
def spawn_proc (f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

# Run multiple `f` in parallel with the list of X
# -----------------------------------------------
def multi_runs (f, X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn_proc(f),args=(c,x)) for (p,c),x in zip(pipe,X)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]
