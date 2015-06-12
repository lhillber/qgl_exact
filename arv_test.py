#!/usr/bin/python

import subprocess

def main(params):
    
    #    subprocess.call(['./tmain.py'])
    for L in params['Llist']:
        for dt in params['dtlist']:
            print(L, dt )



