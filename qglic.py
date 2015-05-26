#!/usr/bin/python

def make_IC(IC):
    ICchars = list(IC)
    if ICchars[0]=='d':
        dec =int(''.join(ICchars[1:]))


make_IC('d3')
