#! /usr/bin/python

import sys, getopt
import arv_test as sim

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
    param_vals = {'batch_path' : 'test',                  \
               'tasks' : ['t','n','nn','MI'],     \
               'Llist' : [10],                    \
              'dtlist' : [1.0],                   \
           'tspanlist' : [(0.0,10.0)],            \
              'IClist' : [[('a',1.0),('W',0.0)]]  \
         }

    
    # use inmput params
    param_vals =get_args(param_vals,argv)
    return param_vals


sim.main(params(sys.argv))
