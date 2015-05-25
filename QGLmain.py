#!/usr/bin/python

import QGLsettings as const
import QGLsim as qsim
import QGLmeasures as qmes
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import numpy as np

const.params()

myoutputs = qmes.outputs()
y = [myoutputs[i]['MI']['ND'] for i in range(const.NSTEPS)]
x = [myoutputs[i]['t'] for i in range(const.NSTEPS)]
plt.plot(x,y)

board = np.array([myoutputs[i]['n']['nexp'] for i in
    range(const.NSTEPS)]).transpose()
plt.matshow(board,fignum = 100,cmap=plt.cm.gray)

plt.show()
