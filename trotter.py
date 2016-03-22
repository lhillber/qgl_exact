#!/usr/bin/python3

from os import makedirs, environ
from os.path import isfile
import numpy as np
import scipy.linalg as sla
from itertools import permutations
import matrix as mx
import states as ss
import fio as fio
import measures as qms


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import qgl_plotting as pt


# default plot font
# -----------------
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)
ops = ss.ops


# totalistic selector for N live sites
# ------------------------------------
def script_N(N, D=4, V='X'):
    perm = list(set([perm for perm in
            permutations(['0']*(D-N) + ['1']*N, 4)]))
    script_N = np.zeros((2**5, 2**5), dtype=complex)
    for tup in perm:
        matlist = [tup[0], tup[1], V, tup[2], tup[3]]
        matlist = [ops[key] for key in matlist]
        script_N += mx.listkron(matlist)
    return script_N

# kronecker together a list of op strings and sum a list of these
# op-string-lists
# ---------------------------------------------------------------
def op_from_list(op_list_list):
    return sum(mx.listkron([ops[key] for key in op_list]) 
               for op_list in op_list_list)

# create Hamiltonians for the bulk and the boundaries
# ---------------------------------------------------
def make_H_list():
    L_list = [['X', '1', '1']]
    l_list = [['0', 'X', '1', '1'],
              ['1', 'X', '0', '1'],
              ['1', 'X', '1', '0'],
              ['1', 'X', '1', '1']]

    r_list = [['1', '1', 'X', '0'],
              ['1', '0', 'X', '1'],
              ['0', '1', 'X', '1'],
              ['1', '1', 'X', '1']]
    R_list = [['1', '1', 'X']]

    HL = op_from_list(L_list)
    Hl = op_from_list(l_list)
    Hj = script_N(2) + script_N(3)
    Hr = op_from_list(r_list)
    HR = op_from_list(R_list)
    H_list = [HL, Hl, Hj, Hr, HR]
    return H_list

# Construct the propigator for a time step dt (hbar=1)
# ----------------------------------------------------
def H_to_U(H, dt):
    return sla.expm(-1j*H*dt)

# Make a dictionary of propigators for the bulk and boundaries
# ------------------------------------------------------------
def make_U_dict(H_list, dt, U_keys = None):
    if U_keys == None:
        U_keys = ['L'+str(dt), 'l'+str(dt), 'j'+str(dt), 
                  'r'+str(dt), 'R'+str(dt)]
    U_list = [H_to_U(H, dt) for H in H_list]
    U_dict = dict(zip(U_keys, U_list))
    return U_dict


# Get the appropriate U and site list for a certian site j
# --------------------------------------------------------
def get_U_js_pair(U_dict, j, L, dt):
    if j == 0:
        U = U_dict['L'+str(dt)]
        js = [0, 1, 2]
    elif j == 1:
        U = U_dict['l'+str(dt)]
        js = [0, 1, 2, 3]
    elif j == L-2:
        U = U_dict['r'+str(dt)]
        js = [L-4, L-3, L-2, L-1]
    elif j == L-1:
        U = U_dict['R'+str(dt)]
        js = [L-3, L-2, L-1]
    else:
        U = U_dict['j'+str(dt)]
        js = [j-2, j-1, j, j+1, j+2]
    return U, js

# Apply the nth layer of the trotter expansion
# --------------------------------------------
def trotter_layer(n, dt, L, state, U_dict):
    for j in range(n, L, 5):
        U, js = get_U_js_pair(U_dict, j, L, dt)
        state = mx.op_on_state(U, js, state)
    return state

def trotter_sym(M, dt, state, U_dict):
    dts = [dt/4, dt/2, dt/4, dt/2, dt/2, dt, dt/2, dt/2, dt/4, dt/2, dt/4]
    ns = [0, 1, 0, 2, 3, 4, 3, 2, 0, 1, 0]
    for m in range(M):
        for n, dt in zip(ns, dts):
            state = trotter_layer(n, dt, L, state, U_dict)
        yield state

def trotter_asym(M, dt, state, U_dict):
    dts = [dt]*5
    for m in range(M):
        for n, dt in zip(range(5), dts):
            state = trotter_layer(n, dt, L, state, U_dict)
        yield state


# Time evolution base on trotter expansion
# ----------------------------------------
def trotter_evolve(state, dt, t0=0, T=1, mode='sym'):
    M = int((T-t0)/dt)
    H_list = make_H_list()
    U_dict = make_U_dict(H_list, dt)
    U_dict.update( make_U_dict(H_list, dt/2, U_keys=['L'+str(dt/2), 'l'+str(dt/2), 'j'+str(dt/2), 
                                                     'r'+str(dt/2), 'R'+str(dt/2)]))
    U_dict.update( make_U_dict(H_list, dt/4, U_keys=['L'+str(dt/4), 'l'+str(dt/4), 'j'+str(dt/4), 
                                                     'r'+str(dt/4), 'R'+str(dt/4)]))
    if mode == 'sym':
        return trotter_sym(M, dt, state, U_dict)

    elif mode == 'asym':
        return trotter_asym(M, dt, state, U_dict)

def measure(M, L, init_state, dt, T=1, mode='sym'):
    nexp = np.zeros((M, L))
    for t, state in enumerate(
            trotter_evolve(init_state, dt, T=T, mode=mode)):
        for j in range(L):
            rtj = mx.rdms(state, [j])
            nexp[t,j] = np.trace(rtj.dot(ss.ops['1'])).real
    return nexp

# helper function for latexed sscientific notation in matplotlib
# --------------------------------------------------------------
def as_sci(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


# Plot the solution as a surface
# ------------------------------
def plot_grid(U, ts, ax, title, nticks=3):
    fig = plt.gcf()


    board = ax.imshow(U, origin='lower', interpolation='none',
            aspect='auto')

    ytick_lbls = ts[::int(len(ts)/nticks)]
    ytick_locs = np.arange(len(ts))[::int(len(ts)/nticks)]
    xtick_lbls = range(L)
    xtick_locs = range(L)
    plt.xticks(xtick_locs,xtick_lbls)
    plt.yticks(ytick_locs,ytick_lbls)


    ax.set_xlabel('Site')
    ax.set_ylabel('Time')
    #ax.set_zlabel(r'$$')
    ax.set_title(title)
    return board
# Plot numeric (U), exact (u) and error (U-u) in three rows of column col
# -----------------------------------------------------------------------
def plot_grids(U, u, dt, col, ts, fignum=1):
    fig = plt.figure(fignum, figsize=(6,9))
    diff = U - u
    data = [U, u, diff] 
    plot_inds = [1, 3, 5]
    data_labels = ['Trotter', 'exact', 'difference']
    for j, dat in enumerate(data):
        title = r'$\Delta t = $' + str(dt) + ': '
        title = title + data_labels[j]
        plot_index = plot_inds[j] + col
        ax = fig.add_subplot(3, 2, plot_index)
        plot_grid(dat, ts, ax, title)
        #if j == 2:
            #ax.set_zlabel('error')



def plot_e_inf(dt_list, e_inf_list, fig, line_num=0):
    # linear fit to loged data
    m, b = np.polyfit(np.log10(dt_list), np.log10(e_inf_list), 1)
    def lin_fit(x):
        x = np.array(x)
        return m*x + b

    # chi squared of fit
    chi2 = np.sum((10**lin_fit(np.log10(dt_list)) - e_inf_list) ** 2)

    # plot
    ax = fig.add_subplot(111)
    ax.plot(dt_list, 10**lin_fit(np.log10(dt_list)),'--r', label='fit')
    ax.loglog(dt_list, e_inf_list,'sk',markersize=5, label='data')

    # label
    ax.set_xlabel(r'$\Delta t$', fontsize=12)
    ax.set_ylabel(r'$e_{\infty,\Delta t}$', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)

    line_num_to_type = {0:'asym',1:'symm'}
    # write fit parameters
    ax.text(0+0.05, 1-(line_num+1)*0.2, 
            line_num_to_type[line_num]+':  slope: {:.3f} \nintercept: {:.3f} \n$\chi^2 = {:s}$'.format(
		   m, b, as_sci(chi2, 3)),
	    transform=ax.transAxes, fontsize=10)

    #ax.set_xlim(right=0.15)
    ax.grid('on')
    return m, b, chi2


def convergence():
    IC = [('c2_f0-1', 1.0)]
    dt_list = [0.1/2**k for k in range(4)]
    output_dir  = 'exact_for_trotter'
    fig = plt.figure(1, figsize=(6.5, 9))
    fig2 = plt.figure(2, figsize=(3.5,3.5))
    for n, mode in enumerate(['asym','sym']):
        e_inf_list = []
        col = 0
        for dt in dt_list:
            exact_data = pt.import_data(output_dir, L, dt, t_span, IC)
            nexp_exact = pt.make_board(exact_data, 'n', 'nexp').T

            M = int(((T - t0)/dt))
            ts = np.arange(t0, T, dt)

            init_state = ss.make_state(L, IC)
            nexp = measure(M, L, init_state, dt, T=T, mode=mode)

            error = nexp - nexp_exact
            e_inf = np.max(np.abs(error)[-1,:])
            e_inf_list.append(e_inf)

            if dt in [dt_list[0], dt_list[3]]:
                plot_grids(nexp, nexp_exact, dt, col, ts, fignum=1)
                col += 1

        fig.subplots_adjust(hspace=0.3, wspace=0.15)
        plot_e_inf(dt_list, e_inf_list, fig2, line_num=n)

    #plt.show()
    fio.multipage('../exact_for_trotter/plots/conv_sample.pdf')

def basic_example(L, t0, T, dt, IC, mode):
    t_span = [t0,T] 
    M = int(((T - t0)/dt))

    # run and measure
    init_state = ss.make_state(L, IC)
    nexp = measure(M, L, init_state, dt, T=T, mode=mode)
    #plot result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts = np.arange(t0, T, dt)
    title = r'$\langle n_j \rangle$, $\Delta t = {:f}$'.format(dt)
    grid = plot_grid(nexp, ts, ax, title, nticks=3)
    plt.colorbar(grid, label= r'$\langle n_j \rangle$')

    plt.show()
   # save plot (comment out the show() to use)
    #fio.multipage('trotter_example.pdf')


if __name__ == '__main__':
    mode = 'sym'
    #mode = 'asym'
    L = 12
    t0 = 0
    T = 2
    dt = 0.01
    IC = 'c2_f0-1'
    basic_example(L, t0, T, dt, IC, mode)
