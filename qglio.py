#!/usr/bin/python
import json
import numpy as np


def write_data(path,data):
    data = np.asarray(data)
    with open(path,'w') as outfile:
        json.dump(data.tolist(),outfile)
    return

def read_data(path):
    with open(path,'r') as infile:
        data = json.load(infile)
    return data

def write_cdata(path,data):
    data = np.asarray(data)
    with open(path,'w') as outfile:
        splitdata=np.array([data.real,data.imag])
        json.dump(splitdata.tolist(),outfile)

def read_cdata(path):
    with open(path,'r') as infile:
        splitdata = np.asarray(json.load(infile))
        data = splitdata[0]+1j*splitdata[1]
    return data
