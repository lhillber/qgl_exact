#!/ur/bin/python 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from math import log,sqrt
import numpy as np
import networkmeasures as nm
import qglopsfuncs as qof
import qglio as qio
import os
import scipy.fftpack as spf
from collections import OrderedDict
font = {'family':'normal', 'weight':'bold', 'size':16}
mpl.rc('font',**font)

def IC_string(IC):
    return '_'.join(['{}_{}'.format(k,v) for k,v in IC.items()])

def sim_name(batch_path,L,dt,tspan,IC,subdir='data'):
    meas_name = 'L'+str(L)+'_dt'+str(dt)+'_tspan'+str(tspan[0])+'-'+str(tspan[1])+'_IC'+IC_string(IC)
    return '../'+batch_path+'/'+subdir+'/'+meas_name

def import_data(batch_path,L,dt,tspan,IC):
    fname = sim_name(batch_path,L,dt,tspan,IC,subdir='data')+'_meas.json'
    mydata = qio.read_data(fname)
    mydata['L']=L
    mydata['dt']=dt
    mydata['tspan']=tspan
    mydata['IC']=IC
    mydata['Nsteps']=round((tspan[1]-tspan[0])/dt)
    return mydata

def import_ham(L):
    ham_path ='../data/hamiltonians/L'+str(L)+'_ham.mtx'
    return sio.mmread(self.ham_path).tocsc()


def make_time_series(mydata,task,subtask):
    time_series = [mydata[task][i][subtask] for i in range(mydata['Nsteps'])]
    times = [mydata['t'][i] for i in range(mydata['Nsteps'])]
    return np.array(times), np.array(time_series)

def make_board(mydata,task,subtask):
    board = np.array([mydata[task][i][subtask] for i in range(mydata['Nsteps'])]).transpose()
    return board

def make_fft(mydata,task,subtask):
    times,time_series = make_time_series(mydata,task,subtask)
    dt = mydata['dt']    
    Nsteps = mydata['Nsteps']
    time_series = time_series - qof.average(time_series)
    if Nsteps%2 == 1:
        time_sereis = np.delete(time_series,-1)
        Nsteps = Nsteps - 1
    amps =  (2.0/Nsteps)*np.abs(spf.fft(time_series)[0:Nsteps/2])
    freqs = np.linspace(0.0,1.0/(2.0*dt),Nsteps/2)
    return freqs, amps


def plot_time_series(mydata,task,subtask,fignum=1,ax=111,yax_label='yax_label',title='title',start=None,end=None):
    start = mydata['tspan'][0] if start is None else start
    end = mydata['tspan'][1] if end is None else end
    times, time_series = make_time_series(mydata,task,subtask)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(times,time_series,'-o')
    plt.xlabel('Time [dt = '+str(mydata['dt'])+']')
    plt.ylabel(yax_label)
    plt.xlim([start,end])
    plt.title(title)
    plt.tight_layout()
    return

def plot_phase_diagram(mydata,task,subtask_x,subtask_y,fignum=1,ax=111,yax_label='subtask_y',xax_label='subtask_x',title='title'):
    time_series_x = make_time_series(mydata,task,subtask_x)[1]
    time_series_y = make_time_series(mydata,task,subtask_y)[1]
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(time_series_x,time_series_y)
    plt.xlabel(xax_label)
    plt.ylabel(yax_label)
    plt.title(title)
    plt.tick_params(axis='both',labelsize=9)
    plt.locator_params(nbins=5)
    plt.tight_layout()
    return

def plot_fft(mydata,task,subtask,fignum=1,ax=111,yax_label='Intensity',title='FFT',start=0,end=5):
    freqs, amps = make_fft(mydata,task,subtask)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    if sum(amps)>len(amps)*1e-14:
        plt.semilogy(freqs,amps,'.')
    else:
        plt.plot(freqs,amps,'.')
    plt.xlabel('Frequency [1/dt]')
    plt.xlim([start,end])
    plt.ylim([amps.min(),10.*amps.max()])
    plt.ylabel(yax_label)
    plt.ylim(1e-5,0.1)
    plt.title(title)
    plt.fill_between(freqs,0,amps)
    plt.tight_layout()
    return

def plot_specgram(mydata,task,subtask,fignum=1,ax=111,yax_label='Frequency',title='title',NFFT=420):
    times, time_series = make_time_series(mydata,task,subtask)
    time_series = time_series - qof.average(time_series)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    
    Pxx, freqs, bin = mlab.specgram(np.array(time_series), \
            window=mlab.window_none, \
            Fs=int(1/(mydata['dt'])), \
            NFFT=NFFT, noverlap=NFFT-1, pad_to=600)
    
    plt.pcolormesh(bin,freqs,Pxx,rasterized=True, cmap=plt.cm.jet)
    cbar=plt.colorbar()
    cbar.set_label('Intensity')
    plt.ylim([0.0,5])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title(title)
    plt.tight_layout()
    return


def time_average(mydata,task,subtask):
    time_series = make_time_series(mydata,task,subtask)[1]
    avg = qof.average(time_series)
    var = np.var(time_series)
    return avg,var


def plot_board(mydata,task,subtask,fignum=1,ax=111,yax_label='Site Number',title='The Quantum Game of Life',start=0,end=None,nticks=5):
    L = mydata['L']
    tmax = mydata['tspan'][1]
    tmin = mydata['tspan'][0]
    dt = mydata['dt']
    Nsteps = mydata['Nsteps']
    end = Nsteps if end is None else end
    Nsteps = end - start
#    xtick_lbls=np.arange(tmin,tmax,dt)[start:end:int(Nsteps/nticks)]
#    xtick_locs=np.arange(0,Nsteps)[start:end:int(Nsteps/nticks)]
#    ytick_lbls=range(1,L-3)
#    ytick_locs=range(2,L-2)
    board = make_board(mydata,task,subtask)
    fig=plt.figure(fignum)
    fig.add_subplot(ax)
    if subtask=='nexp':
        plt.imshow(board,
                vmin=0.,
                vmax=1.,
                origin='lower',
                cmap=plt.cm.jet,
                interpolation='none',
                aspect='auto',
                rasterized=True)
    elif subtask=='DIS':
        cmap=mpl.colors.ListedColormap([plt.cm.jet(0.),plt.cm.jet(1.)])
        bounds=[0,0.5,1]
        norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
        plt.imshow(board,
                vmin=0.,
                vmax=1.,
                origin='lower',
                cmap=cmap,
                norm=norm,
                interpolation='none',
                aspect='auto',
                rasterized=True)
#    plt.xticks(xtick_locs,xtick_lbls)
#    plt.yticks(ytick_locs,ytick_lbls)
#    plt.ylim(2,L-3) 
    cbar=plt.colorbar()
    cbar.set_label(r'$\langle n_i \rangle$')
    plt.xlabel('Time [dt = '+str(mydata['dt'])+']')
    plt.ylabel(yax_label)
    plt.title(title)
    plt.tight_layout()
    return


def multipage(fname, figs=None, clf=True, dpi=300): 
    pp = PdfPages(fname) 
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp,format='pdf')
        
        if clf==True:
            fig.clf()
    pp.close()
    return


def make_tave_sweep(batch_path,L,dt,tspan,IClist,task,subtask):
    avelist = []
    varlist = []
    for IC in IClist:
        IC = OrderedDict(IC)
        mydata = import_data(batch_path,L,dt,tspan,IC)
        ave,var = time_average(mydata,task,subtask)
        avelist.append(ave)
        varlist.append(var)
    return avelist, varlist

def plot_tave_sweep(batch_path,L,dt,tspan,IClist,task,subtask,thlist,fignum=1,ax=111,yax_label='Equilibrium val.',title='title'):
    avelist,varlist = make_tave_sweep(batch_path,L,dt,tspan,IClist,task,subtask) 
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    fmt = '-o'
    if subtask == 'DIV':
        varlist = [0]*len(thlist)
    plt.errorbar(thlist, avelist, yerr=varlist, fmt=fmt)
    plt.xlabel('Mixing Angle [rad]')
    plt.ylabel(yax_label)
    plt.title(title)
    plt.tight_layout()
    return

def time_plots(mydata):
    plot_board(mydata,'n','nexp',ax=211,fignum=1)
    plot_board(mydata,'n','DIS',ax=212,fignum=1,title='Discretized QGL')

    plot_time_series(mydata,'n','DEN',fignum=2,ax=211,yax_label=r'$\rho$',title='')
    plot_fft(mydata,'n','DEN',fignum=2,ax=212)

    plot_time_series(mydata,'n','DIV',fignum=3,ax=211,yax_label=r'$\Delta$',title=r'')

    plot_time_series(mydata,'nn','CC',fignum=4,ax=211,yax_label='CC',title=r'$g_{ij}$')
    plot_fft(mydata,'nn','CC',fignum=4,ax=212)
    plot_time_series(mydata,'nn','ND', fignum=5, ax=211,yax_label='ND',title=r'$g_{ij}$')
    plot_fft(mydata,'nn','ND', fignum=5, ax=212)
    plot_time_series(mydata,'nn','Y', fignum=6, ax=211,yax_label='Y',title=r'$g_{ij}$')
    plot_fft(mydata,'nn','Y', fignum=6, ax=212)
    plot_time_series(mydata,'nn','HL',fignum=7, ax=211,yax_label='IHL',title=r'$g_{ij}$')
    plot_fft(mydata,'nn','HL', fignum=7, ax=212)

    plot_specgram(mydata,'nn','CC',fignum=8,ax=111,title=r'$g_{ij}$ CC Spectrogram')
    plot_specgram(mydata,'nn','ND',fignum=9,ax=111,title=r'$g_{ij}$ ND Spectrogram')
    plot_specgram(mydata,'nn','Y',fignum=10,ax=111,title=r'$g_{ij}$ Y Spectrogram')
    plot_specgram(mydata,'nn','HL',fignum=11,ax=111,title=r'$g_{ij}$ IHL Spectrogram')

    plot_phase_diagram(mydata,'nn','CC','ND',fignum=12,ax=231,xax_label='CC',yax_label='ND',title = r'$g_{ij}$')
    plot_phase_diagram(mydata,'nn','CC','Y',fignum=12,ax=232,xax_label='CC',yax_label='Y',title = r'$g_{ij}$')
    plot_phase_diagram(mydata,'nn','CC','HL',fignum=12,ax=233,xax_label='CC',yax_label='IHL',title = r'$g_{ij}$')
    plot_phase_diagram(mydata,'nn','ND','Y',fignum=12,ax=234,xax_label='ND',yax_label='Y',title = r'$g_{ij}$')
    plot_phase_diagram(mydata,'nn','ND','HL',fignum=12,ax=235,xax_label='ND',yax_label='IHL',title = r'$g_{ij}$')
    plot_phase_diagram(mydata,'nn','Y','HL',fignum=12,ax=236,xax_label='Y',yax_label='IHL',title = r'$g_{ij}$')

    plot_time_series(mydata,'MI','CC',fignum=13,ax=211,yax_label='CC', title=r'$\mathcal{I}_{ij}$')
    plot_fft(mydata,'MI','CC',fignum=13, ax=212)
    plot_time_series(mydata,'MI','ND', fignum=14, ax=211,yax_label='ND',title=r'$\mathcal{I}_{ij}$')
    plot_fft(mydata,'MI','ND', fignum=14, ax=212)
    plot_time_series(mydata,'MI','Y', fignum=15, ax=211,yax_label='Y',title=r'$\mathcal{I}_{ij}$')
    plot_fft(mydata,'MI','Y', fignum=15, ax=212)
    plot_time_series(mydata,'MI','HL',fignum=16, ax=211,yax_label='IHL',title=r'$\mathcal{I}_{ij}$')
    plot_fft(mydata,'MI','HL', fignum=16, ax=212)

    plot_specgram(mydata,'nn','CC',fignum=17,ax=111,title=r'$\mathcal{I}_{ij}$ CC Spectrogram')
    plot_specgram(mydata,'nn','ND',fignum=18,ax=111,title=r'$\mathcal{I}_{ij}$ ND Spectrogram')
    plot_specgram(mydata,'nn','Y',fignum=19,ax=111,title=r'$\mathcal{I}_{ij}$ Y Spectrogram')
    plot_specgram(mydata,'nn','HL',fignum=20,ax=111,title=r'$\mathcal{I}_{ij}$ IHL Spectrogram')

    plot_phase_diagram(mydata,'MI','CC','ND',fignum=21,ax=231,xax_label='CC',yax_label='ND',title = r'$\mathcal{I}_{ij}$')
    plot_phase_diagram(mydata,'MI','CC','Y',fignum=21,ax=232,xax_label='CC',yax_label='Y',title = r'$\mathcal{I}_{ij}$')
    plot_phase_diagram(mydata,'MI','CC','HL',fignum=21,ax=233,xax_label='CC',yax_label='IHL',title =r'$\mathcal{I}_{ij}$')
    plot_phase_diagram(mydata,'MI','ND','Y',fignum=21,ax=234,xax_label='ND',yax_label='Y',title = r'$\mathcal{I}_{ij}$')
    plot_phase_diagram(mydata,'MI','ND','HL',fignum=21,ax=235,xax_label='ND',yax_label='IHL',title = r'$\mathcal{I}_{ij}$')
    plot_phase_diagram(mydata,'MI','Y','HL',fignum=21,ax=236,xax_label='Y',yax_label='IHL',title = r'$\mathcal{I}_{ij}$')
    plt.tight_layout()
    return

def theta_plots(batch_path,L,dt,tspan,IClist,thlist):
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'n','DEN',thlist,fignum=22,ax=111,yax_label=r'$\rho_{\infty}$',title=r'Equilibrium $\rho$')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'n','DIV',thlist,fignum=23,ax=111,yax_label=r'$\Delta_{\infty}$',title=r'Equilibrium $\Delta$')
   
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'MI','CC',thlist,fignum=24,ax=111,yax_label=r'CC$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium CC')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'MI','ND',thlist,fignum=25,ax=111,yax_label=r'ND$_{\infth}$',title=r'$\mathcal{I}_{ij}$ Equilibrium ND')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'MI','Y',thlist,fignum=26,ax=111,yax_label=r'Y$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium Y')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'MI','HL',thlist,fignum=27,ax=111,yax_label=r'IHL$_{\infty}$',title=r'$\mathcal{I}_{ij}$ Equilibrium IHL')
   
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'nn','CC',thlist,fignum=28,ax=111,yax_label=r'CC$_{\infty}$',title=r'g$_{ij}$ Equilibrium CC')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'nn','ND',thlist,fignum=29,ax=111,yax_label=r'ND$_{\infty}$',title=r'g$_{ij}$ Equilibrium ND')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'nn','Y',thlist,fignum=30,ax=111,yax_label=r'Y$_{\infty}$',title=r'g$_{ij}$ Equilibrium Y')
   plot_tave_sweep(batch_path,L,dt,tspan,IClist,'nn','HL',thlist,fignum=31,ax=111,yax_label=r'IHL$_{\infty}$',title=r'g$_{ij}$ Equilibrium IHL')
   plt.tight_layout() 

def main(batch_path,Llist,dtlist,tspanlist,IClist,thlist):
    for L in Llist:
        for dt in dtlist:
            for tspan in tspanlist:
#                theta_plots(batch_path,L,dt,tspan,IClist,thlist)
#                multipage('../'+batch_path+'/plots/'+'equibvals.pdf',clf=False)
#                '''
                for ic in IClist[::8]:
                    IC = OrderedDict(ic)
                    mydata = import_data(batch_path,L,dt,tspan,IC) 
                    plt.close()
                    time_plots(mydata)
                    multipage(sim_name(batch_path,L,dt,tspan,IC,subdir='plots')+'.pdf')
                    plt.close()
#                '''

