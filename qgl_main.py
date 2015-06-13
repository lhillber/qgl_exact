#!/usr/bin/python

import qgl_config as config
import qgl_sim2 as qsim
import sys, getopt

def get_args(param_vals,argv):
    
    param_names = [el + '=' for el in param_vals.keys()]
    
    try:
        opts, args = getopt.getopt(argv[1:],"",param_names)
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        opt = opt[2:]

        if opt in param_vals.keys():
            param_vals[opt] = eval(arg)

    return param_vals


def params(argv):
    
    #set defaults
    param_vals = config.params
                 
    # use inmput params
    param_vals = get_args(param_vals,argv)
    return param_vals



qsim.main(params(sys.argv))