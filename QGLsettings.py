#!/usr/bin/python
import numpy as np
# simulation parameters
def params():
    global L,INIT,BC,TMAX,DT,DT_STRING,NSTEPS,SIM_NAME,DATA_PATH,HAM_PATH,OPS
    L = 10
    BC = ['dead','dead','dead','dead']
    INIT = 3
    TMAX = 10
    DT = 0.1
    DT_STRING = '01'
    NSTEPS = int(round(TMAX/DT)+1)
    SIM_NAME = 'L'+str(L)+'_tmax'+str(TMAX)+'_dt'+DT_STRING+'_'
    DATA_PATH = '../data/'
    HAM_PATH = '../data/Hamiltonians/'
    OPS = {'I':np.array([[1.,0.],[0.,1.]]),'n':np.array([[0.,0.],[0.,1.]]),'nbar':np.array([[1.,0.],[0.,0.]]),'mix':np.array([[0.,1.],[1.,0.]]),'dead':np.array([[1.,0.]]).transpose(),'alive':np.array([[0.,1.]]).transpose()}
