#script to process run and save matrix
#capability to generate submatrices for bootstrap error analysis

import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *
import random
from tqdm import tqdm

plt.style.use('style')

jacobian = False
save = True
plot = False
bootstrap = False

data_path = '/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/data'
out_folder = 'processed_data_v3'

run = 275

dir_name = '{}/Run_{}'.format(out_folder,run)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('directory created')
else:
    print('directory already exists, will overwrite if saving...')

def init_mat(run,N):
    df = make_runmap('data/Data Log.tsv')
    t0 = df[df.run==run].t0.values
    fpath = get_fpath(data_path,str(run))[0]

    fname = os.path.basename(fpath)
    tof = get_tof_axis(fpath)
    df = get_delay_data(fpath,N,t0)

    try:
        params = np.loadtxt('calibrations/{}_calib.txt'.format(fname))
    except:
        raise Exception('No calibration file found for {}'.format(fname))

    x = tof2energy(tof,*params) #set so data doesn't calibrate?
    y = np.array(df.delay)
    z = np.vstack(df.pes)

    if jacobian:
        for idx in range(len(z)):
            z[idx] = jacob(tof, z[idx], *params)

    mask = x>0

    x = x[mask]
    z = z[:,mask]

    idx = np.argmin(x)
    x = x[idx:]
    z = z[:,idx:]

    return x,y,z


x,y,z = init_mat(run,1152)

if save:
    mat = np.pad(z,((1,0),(1,0)),mode = 'constant')
    mat[1:,0] = y
    mat[0,1:] = x
    np.savetxt('{}/main.txt'.format(dir_name,run),mat)


if plot:
    plt.figure(figsize = (3,3))
    plt.pcolormesh(x,y,z) 
    plt.colorbar()
    plt.xlim(3,10)
    plt.xlabel('Energy / eV')
    plt.ylabel('Delay / fs')

##bootstrap function##

if bootstrap:
    for l in tqdm(range(0,20)):
        cycles = range(100,1200,100)
        total = []

        for i, N in enumerate(cycles):

            if i == 0:
                x,y,z = init_mat(run,N)

            else:
                x,y,z = init_mat(run,N)
                z = z - init_mat(run,cycles[i-1])[2]

            total.append(z)

        sampled_total = random.choices(total,k = len(total))
        sampled_sum = np.sum(sampled_total,axis = 0)

        mat = np.pad(sampled_sum,((1,0),(1,0)),mode = 'constant')
        mat[1:,0] = y
        mat[0,1:] = x

        np.savetxt('{}/submatrix_{}.txt'.format(dir_name,l),mat)
    #save summed matrix






# init_mat(run,1000)[2]-init_mat(run,500)[2]






#load corresponding calibration file



