#!/usr/bin/python3
import numpy as np

def visibility(vals):
    #Basic function for computing visibility given a list.
    return np.abs(np.max(vals)-np.min(vals))
def visibility2(i,t,T):
    #Grab number operator data for site i from times t' such that t-T/2 <= t' <= t+T/2 from MPS and feed into visibility.  Store in a list.

    #Return the list
    return

def visibility3(t,T):
    #Iterate visibility2 over lattice site.
    return

def visibility4(T):
    #Iterate visibility3 over times t'.
    return

#supposing I have a timelist, lets assume equally spaced dt.
#Once we determine what T is, 
dt=0.1
timelist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
timelistnp=np.array(timelist)
def visibile2(t,T):
    #Computes the visibility of all sites across the lattice,
    #at time t, and with time ranging from t-T/2 to t+T/2.
    #Endpoints are included per the definition given by:
    #doi:10.1209/0295-5075/97/20012
    matrix=[]
    timelogic=np.logical_and(np.array(timelist)<=t+(T/2),t-(T/2)<=np.array(timelist))
    timspan=timelist[timelogic]
    for time in timespan:
        matrix.append((mps.GetObservable(Outputs,'time',time))['n'])
    matrix2=np.array(matrix)
    maxvalues=np.max(matrix2,axis=0)
    minvalues=np.min(matrix2,axis=0)
    vis=np.abs(maxvalues-minvalues)
    return vis


def D(val):
    #Assuming val is between zero and one, this rounds to one if val is greater than 0.5 and rounds to zero otherwise.
    np.round(val)
    return
def Discrete(values):
    #Map D over the list of number operators received.
    return map(D,values)

def density(discrete):
    #Iterate Discrete(i,t) over site index. Divide by L
    np.mean(discrete)
    return

#Pieced together from:
#http://stackoverflow.com/questions/14529523/python-split-for-lists
#from itertools import groupby
#[list(group) for k, group in groupby(a,bool) if k]
import itertools
def cluster(values):
    clustdict={}
    #Returns dictionary of s:n(s)
    #Assumes import itertools has been placed earlier in the python notebook.
    #"A cluster is a set of live sites connected by a first-nearest-neighbour relation."
    #http://dx.doi.org/10.1088/0305-4470/26/22/019
    #Given a list of bits, cluster(lattice,s,t) will compute the number of clusters of length s
    #This returns a list of contiguous chains of alive cells.
    chains=[list(group) for k, group in itertools.groupby(values,bool) if k]
    chainlens=list(map(len,chains))
    unique = list(set(chainlens))
    for elem in unique: clustdict[elem]=chainlens.count(elem)
    return clustdict
#1
def density(values):
    totalalive=float(np.sum(values))
    totalatall=len(values)
    return totalalive/totalatall
"""
def div(clustersizes):
    #Grab the keys
    divtmp=len(clustersizes)
    return divtmp
"""
#2
def diversity(clusterdict):
    clustkeys=clusterdict.keys()
    div=len(clustkeys)
    return div


#print cluster([1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1])
#Can append this dictionary to a larger dictionary
#This can be simplied, consult manual.
#by Output[]['clusters']=cluster()
#timeslice={}
#timesliced[t]=mps.GetObservables(Outputs ,’t ’,t)
#timesliced[t]['Di']=Discrete[timesliced[t]['n']]
#timesliced[t]['clusters']=cluster(timesliced[t]['Di'])







