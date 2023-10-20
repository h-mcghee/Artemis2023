import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *

plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Style_files/style')

tof,df,fname = process_energy(275)
tof2,df2,fname2 = process_energy(289)

# combine datasets - could maybe write a function for this?
total = pd.concat([df,df2])
total.delay = total.delay.apply(lambda x: x[0]) # takes values out of square brackets - might need to change from functions
df = total.groupby('delay',as_index = False).sum()

# tof,df,fname = process_energy(305)

save = False

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


plt.figure(figsize = (3,3))
plt.pcolormesh(y,x,z.T,vmin = 0,vmax = 50)
plt.colorbar()
# plt.xlim(-100,1000)
plt.ylim(3,9.5)
plt.xlabel('Delay / fs')
plt.ylabel('eBE / eV')
plt.gca().invert_yaxis()

plt.show()

plt.figure(figsize = (6,3))

tof_window = [3585,3596]
tof_window = [3596,3601]
tof_window = [3610,3619]
tof_window = [3621,3623]
tof_window = [3632,3635]

# vlineout = [10,12.0]
vlineout = tof2energy(tof_window,*params)
# vlineout = [9.1,10.3]
# vlineout = [6,8]
hlineout = [5000,10000]

h = np.mean(z[:,tof_slice(x,vlineout[0],vlineout[1])],axis=1)
plt.plot(y,h,label = '{:.1f}-{:.1f} eV'.format(vlineout[0],vlineout[1]))

plt.xlabel('Delay / fs')
plt.ylabel('Counts')
plt.legend()
plt.xscale('symlog',linthreshx = 1500)
if save:
    np.savetxt('processed_data/trans_79V_delay_{:.1f}-{:.1f}eV.txt'.format(vlineout[0],vlineout[1]),np.c_[y,np.mean(z[:,tof_slice(x,vlineout[0],vlineout[1])],axis=1)])
plt.show()

plt.figure(figsize = (6,3))
plt.plot(x,np.mean(z[tof_slice(y,hlineout[0],hlineout[1]),:],axis=0),label = '{}-{} fs'.format(hlineout[0],hlineout[1]))
plt.xlabel('ToF / eV')
plt.ylabel('Counts')
# plt.ylim(0,30)
# plt.xlim(0,9.5)
plt.legend()
plt.show()


