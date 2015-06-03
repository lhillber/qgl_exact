#!/usr/bin/python

from math import cos, sin
from numpy import linspace, pi

import qgl_sim as qsim
import qgl_plotting as qplt


postprocessing=False
batch_path = 'WvA2'
tasks = ['t','n','nn','MI']
Llist = [7,8,9]
dtlist = [0.01]
tspanlist= [(1000,1020)]
thlist = linspace(0,pi/2,20)
IClist = [[('W',cs),('a',ss)] for cs,ss in [(cos(th),sin(th)) for th in thlist]]
#IClist = [[('W',1.)],[('G',1.)],[('d12',1.)],[('a',1.)]]


if not postprocessing:
    qsim.main(batch_path,tasks,Llist,dtlist,tspanlist,IClist)
elif  postprocessing:
    qplt.main(batch_path,Llist,dtlist,tspanlist,IClist,thlist)
