import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *
import random
from tqdm import tqdm

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

#takes 45s to sample 8 spectra with replacement 5 time

run = 275
reset = False

# plt.figure(figsize = (6,3))

eBE = [3.0,7.5] 

plt.figure(figsize = (6,3))

if reset:

    long_list = []
    for loop in range(0,10):
        cycles = range(20,1160,20)
        total = []

        for i, N in tqdm(enumerate(cycles)):

            if i == 0:
                tof,df,fname = process_energy(run,N)

            else:
                tof,df,fname = process_energy(run,N)
                dfH = process_energy(run,cycles[i-1])[1]
                df.pes = df.pes - dfH.pes

            total.append(df)

        sampled_total = random.choices(total,k = len(total))

        df_concat = pd.concat(sampled_total)
        df_concat.delay = df_concat.delay.apply(lambda x: x[0]) # takes values out of square brackets - might need to change from functions
        df_sum = df_concat.groupby('delay',as_index = False).sum()


        try:
            params = np.loadtxt('calibrations/{}_calib.txt'.format(fname))
        except:
            raise Exception('No calibration file found for {}'.format(fname))



        long_list.append(df_sum)

else:
    pass


delay_list = []
for df in long_list:

    # df = df_sum

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
    bkg = np.sum(bkg_matrix[:,tof_slice(x,eBE[0],eBE[1])],axis=1)

    plt.errorbar(y,h-bkg, label = '{}'.format(N))
    delay_list.append(h-bkg)
    plt.xlabel('Delay / fs')
    plt.ylabel('Counts')
    plt.legend()
    plt.xlim(-1000,2000)

error = np.std(delay_list,axis = 0)

plt.figure()
tof,df,fname = process_energy(run,N = 1152)

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
bkg = np.sum(bkg_matrix[:,tof_slice(x,eBE[0],eBE[1])],axis=1)


plt.errorbar(y,h-bkg, yerr = error,label = '{}'.format(N))
plt.xlabel('Delay / fs')
plt.ylabel('Counts')
plt.legend()
# plt.xlim(-1000,2000)
# plt.axhline(0,color = 'k',linestyle = '--')

# plt.ylim(-1000,1000)

# plt.show()





