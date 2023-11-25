import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *

plt.style.use('style')


def process_energy(run,N):
    df = make_runmap('data/Data Log.tsv')
    t0 = df[df.run==run].t0.values
    # N = df[df.run==run].sweeps.values -1

    fpath = get_fpath(data_path,str(run))[0]
    fname = os.path.basename(fpath)

    tof = get_tof_axis(fpath)

    df = get_delay_data(fpath,N,t0)

    return tof,df,fname


run = 305

# plt.figure(figsize = (6,3))

eBE = [11.8,12.0] 

for N in [1000,1200,1300,1400,1500,1660]:
    plt.figure(figsize = (6,3))
    tof,df,fname = process_energy(run,N)

    try:
        params = np.loadtxt('calibrations/{}_calib.txt'.format(fname))
    except:
        raise Exception('No calibration file found for {}'.format(fname))

    x = tof2energy(tof,*params)
    y = np.array(df.delay)
    z = np.vstack(df.pes)

    mask = x>0

    x = x[mask]
    z = z[:,mask]

    idx = np.argmin(x)
    x = x[idx:]
    z = z[:,idx:]

    bkg = tof_slice(y,-1500,-300)
    bkg_matrix = np.mean(z[bkg,:],axis = 0)[None,:]

    h = np.sum(z[:,tof_slice(x,eBE[0],eBE[1])],axis=1) 
    h_error = np.sqrt(h)
    bkg = np.sum(bkg_matrix[:,tof_slice(x,eBE[0],eBE[1])],axis=1)
    bkg_error = np.sqrt(bkg)
    error = np.sqrt(h_error**2 + bkg_error**2)

    plt.errorbar(y,h-bkg,error, label = '{}'.format(N))
    plt.xlabel('Delay / fs')
    plt.ylabel('Counts')
    plt.legend()
    plt.xlim(-1000,2000)
    plt.ylim(-1000,1000)

plt.show()





