#!/usr/bin/python

import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations


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

# Accessor (should just use the global OPS)
# -----------------------------------------
def ops(key):
    return OPS[key]


# Matrix functions
# ================

# Kroeneker product list of matrices
# ----------------------------------
def matkron(matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)

# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron(matlist):
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))

# Hermitian conjugate
# -------------------
def dagger(mat):
    return mat.conj().transpose()

# List average (should replace with np.mean)
# ------------------------------------------
def average(mylist):
    return sum(mylist)/len(mylist)


# Parallelize
# ===========
# http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class

# Wrapper for multiprocessing
# ---------------------------
def spawn_proc(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

# Run multiple `f` in parallel with the list of X
# -----------------------------------------------
def multi_runs(f, X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn_proc(f),args=(c,x)) for (p,c),x in zip(pipe,X)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]
