#!/usr/bin/python

def make_IC(IC):
    ICchars = list(IC)
    if ICchars[0]=='d':
        print(int(ICchars[1:]))

make_IC('d11')
