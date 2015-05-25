#!/usr/bin/python
import numpy as np
import networkx as nx
#Scripted by David Vargas
#---------------------------------------------
def density(matrix):
    #Calculates density, also termed connectance in some
    #literature. Defined on page 134 of Mark Newman's book
    #on networks.
    l=len(matrix)
    lsq=l*(l-1)
    return sum(sum(matrix))/lsq
def clustering(matrix):
    #Calculates the clustering coefficient
    #as it is define in equation 7.39 of
    #Mark Newman's book on networks. Page 199.
    l=len(matrix)
    matrixcube=np.linalg.matrix_power(matrix,3)
    matrixsq=np.linalg.matrix_power(matrix,2)
    #Zero out diagonal entries. So we do not count edges as
    #connected triples.
    for i in range(len(matrixsq)):
        matrixsq[i][i]=0
    denominator=sum(sum(matrixsq))
    numerator=np.trace(matrixcube)
    return numerator/denominator
def disparity(matrix):
    #Calculates the average disparity of a network under
    #the assumption that it is completely connected.
    #Disparity defined on page 199 of doi:10.1016/j.physrep.2005.10.009 
    #Equation 2.39
    l=len(matrix)
    numerator=sum(matrix**2)/l
    denominator=sum(matrix)**2
    return sum(numerator/denominator)

#NetworkX additions
#---------------------------------------------
def distance(mutualinformation):
    #Initialize array
    length=len(mutualinformation)
    thisdistance=np.zeros((length,length))
    #If an element has value zero, set its distance
    #to 10^16. Otherwise set it mij^(-1).
    for i in range(length):
        for j in range(length):
            if mutualinformation[i,j]==0:
                thisdistance[i,j]=np.power(10.,16)
            else:
                thisdistance[i,j]=np.power(mutualinformation[i,j],-1)
    return thisdistance
def geodesic(distance):
    #Initialize networkx graph object
    latticelength=len(distance)
    G=nx.Graph(distance)
    #Use networkx algorithm to compute the shortest path from
    #the first lattice site to the last lattice site.
    pathlength=nx.shortest_path_length(G,0,latticelength-1,weight='weight')
    if pathlength>np.power(10.,15):
        return np.nan
    return pathlength
