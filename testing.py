#!/usr/bin/python


def symetrize_fock(L, dec):
    b = '{0:0'+str(L)+'b}'
    b = b.format(dec)
    print(b)
    b = int(''.join(list(reversed(b))),2)
    print(b)

symetrize_fock(5,3)
