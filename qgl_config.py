#!/usr/bin/python

from math import cos, sin
from numpy import linspace, pi

import qgl_sim2 as qsim
import qglplotting as qplt

'''
postprocessing=True
batch_path = 'AWG12'
tasks = ['t','n','nn','MI']
Llist = [12]
dtlist = [0.1]
tspanlist = [(0,10)]
thlist = linspace(0,pi/2,25)
IClist = [[('a',1.)],[('W',1.)],[('G',1.)]]
'''

params = {'batch_path' : 'test',                  \
               'tasks' : ['t','n','nn','MI'],     \
               'Llist' : [10],                    \
              'dtlist' : [1.0],                   \
           'tspanlist' : [(0.0,10.0)],            \
              'IClist' : [[('a',1.0),('W',0.0)]]  \
         }


'''
postprocessing=True
batch_path = 'WvA6'
tasks = ['t','n','nn','MI']
Llist = [8,9,10,11,12]
dtlist = [0.05]
tspanlist= [(2000,2025)]
thlist = linspace(0,pi/2,25)
IClist = [[('W',cs),('a',ss)] for cs,ss in [(cos(th),sin(th)) for th in thlist]]
'''

'''
postprocessing=True
batch_path = 'WvA'
tasks = ['t','n','nn','MI']
Llist = [7,8,9]
dtlist = [0.01]
tspanlist= [(1000,1010)]
thlist = linspace(0,pi/2,7)
IClist = [[('W',cs),('a',ss)] for cs,ss in [(cos(th),sin(th)) for th in thlist]]
'''
'''
postprocessing = False
batch_path = 'GWtest2'
tasks = ['t','n','nn','MI']
Llist = [7]
dtlist = [0.01]
tspanlist= [(1000,1010)]
thlist = linspace(0,pi/2,7)
IClist = [[('W',1.)],[('G',1.)],[('d12',1.)],[('a',1.)]]
'''

#uuIClist = [[('W',cs),('a',ss)] for cs,ss in [(cos(th),sin(th)) for th in thlist]]



#IClist = [[('W',1.)],[('G',1.)],[('d12',1.)],[('a',1.)]]

'''
if not postprocessing:
    qsim.main(batch_path,tasks,Llist,dtlist,tspanlist,IClist)
elif  postprocessing:
    qplt.main(batch_path,Llist,dtlist,tspanlist,IClist,thlist)
'''
