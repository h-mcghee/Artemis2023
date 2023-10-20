import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *

plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Style_files/style')

fig,(ax1,ax2) = plt.subplots(2,figsize = (3,3))

tof_window = [3570,3596]
tof_window = [3596,3601]
tof_window = [3610,3619]
tof_window = [3621,3623]
tof_window = [3632,3635]

for run in [275,305]:

    tof,df,fname = process_energy(run)

    #load corresponding calibraiton file
    try:
        params = np.loadtxt('calibrations/{}_calib.txt'.format(fname))
    except:
        raise Exception('No calibration file found for {}'.format(fname))

    x = tof2energy(tof,*params)

    y = np.array(df.delay)
    z = np.vstack(df.pes)

#might be a better way to do this...
    mask = x>0

    x = x[mask]
    z = z[:,mask]

    idx = np.argmin(x)
    x = x[idx:]
    z = z[:,idx:]

#bkg sub
    bkg_sub = True

    bkg = tof_slice(y,-1500,-300)
    gs = np.mean(z[bkg,:],axis = 0)

    if bkg_sub:
        z = z - np.mean(z[bkg,:],axis = 0)[None,:]
    else:
        pass


    #plot groundstate
    
    ax1.plot(x,gs/np.max(gs),label = run)

    vlineout = tof2energy(tof_window,*params)

    ax2.plot(y,np.mean(z[:,tof_slice(x,vlineout[0],vlineout[1])],axis=1),label = '{:.1f}-{:.1f} eV'.format(vlineout[0],vlineout[1]))
    ax2.set_xlabel('Delay / fs')
    ax2.set_ylabel('Counts')
    ax2.legend()


plt.tight_layout()
plt.show()



# plt.figure(figsize = (6,3))
# plt.plot(x,np.mean(z[tof_slice(y,hlineout[0],hlineout[1]),:],axis=0),label = '{}-{} fs'.format(hlineout[0],hlineout[1]))
# plt.xlabel('ToF / eV')
# plt.ylabel('Counts')
# # plt.ylim(0,30)
# # plt.xlim(0,9.5)
# plt.legend()
# plt.show()


