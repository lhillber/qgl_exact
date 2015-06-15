#!/usr/bin/python

import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations


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
