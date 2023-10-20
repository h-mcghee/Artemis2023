#Given a tof spectrum, this script will fit the calibration curve and output the parameters
#Calibration file is saved in Calibrations


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit import Parameters
import pandas as pd

from calib_functions import tof2energy
from calib_functions import jacob
from calib_functions import energy2tof,tof2eKE
from functions import tof_slice, process_energy

plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Style_files/style')

#Plot initial data

tof,df,fname = process_energy(305)

# tof,df,fname = process_energy(275)
# tof2,df2,fname2 = process_energy(289)

# # combine datasets - could maybe write a function for this?
# total = pd.concat([df,df2])
# total.delay = total.delay.apply(lambda x: x[0]) # takes values out of square brackets - might need to change from functions
# df = total.groupby('delay',as_index = False).sum()


x = tof
y = df[df.delay < -200].pes.sum()

#fit calibration using tof2energy function (in calib_functions)

tof_cal = [3605,3621,3629,3642,3670,3700,3772]
energy_cal = [9.73,11.8,12.64,13.77,15.6,16.98,18.78]

# tof_cal = [3605,3622,3629,3642]#,3670,3700,3772]
# energy_cal = [9.8,11.9,12.1,12.7]#,15.6,16.98,18.78]

model = Model(tof2energy)

params = Parameters()

params.add('hv', value=21.2, min=20,max=23,vary = True) #in eV
params.add('s', value=0.3, min=0.25,vary = True)    #in m
params.add('t0', value=3400, min=3000,vary = True)  # in ns
params.add('E0', value=0,vary = False)  #in V

result = model.fit(energy_cal, params, x=tof_cal)

#generate array of parameters 

params = []
for param_name in result.params:
    params.append(result.params[param_name].value)

#get rid of negative values

t0 = result.params['t0'].value
y = y[(x<4000) & (x>t0)]
x = x[(x<4000) & (x>t0)]

#plot results

plt.figure(figsize = (6,3))
plt.plot(x,y,'k',label = fname)
plt.legend()
plt.show()

plt.figure(figsize = (6,3))
plt.plot(tof_cal,energy_cal,'ko')
plt.plot(tof_cal,result.best_fit,'r-')
plt.legend()
plt.show()


plt.figure(figsize = (6,3))
plt.plot(tof2energy(x,*params),y,'k')
plt.xlim(0,22)
plt.legend()
plt.show()


plt.figure(figsize=(3,3))
# plt.plot(tof2energy(x,*params),jacob(x,y,params),'r')
mask = tof2energy(x,*params) > 0
plt.plot(tof2energy(x,*params)[mask],jacob(x,y,params)[mask]/np.max(jacob(x,y,params)[mask]),'k')
# np.savetxt('cis_gs.txt',np.c_[tof2energy(x,*params)[mask],jacob(x,y,params)[mask]/np.max(jacob(x,y,params)[mask])])
plt.xlim(0,22)
# plt.ylim(0,0.02e-30)
plt.legend()
plt.show()

fig,ax=plt.subplots(figsize=(3,3))

ax.plot(tof2energy(x,*params),jacob(x,y,params)/1.75e-30,'r',label = 'Jac corrected')
ax.plot(tof2energy(x,*params),y,'k',label = 'uncorrected')

# np.savetxt('cis_79_gs_calib',np.c_[tof2energy(x,*params),jacob(x,y,result)/1.75e-38])

ax.tick_params(axis='x',top = False)
ax.minorticks_off()

ax.set_xlabel('Energy / eV')
ax.set_xlim(8,18)

# plt.ylim(0,0.02e-36)
plt.ylim(-0.1,1.2)

a = lambda x: tof2energy(x,*params)
b = lambda x: energy2tof(x,*params)

ax2 = ax.secondary_xaxis('top', functions=(b,a))
ax2.set_xlabel('ToF / ns')
ax2.set_xticks([3600,3625,3650,3700,3800])
ax2.minorticks_off()

plt.legend()
plt.show()

# np.savetxt('calibrations/{}_calib.txt'.format(fname),params)


# plt.figure(figsize=(3,3))
# cis = np.genfromtxt('txt_files/cis_79_gs_calib.txt')
# trans = np.genfromtxt('txt_files/trans_79_gs_calib.txt')
# label = ['trans','cis']
# for l,i in enumerate([trans,cis]):
#     plt.plot(i[:,0],i[:,1],label = label[l])
# plt.legend()
# plt.xlim(8,20)
# plt.ylim(-0.1,1.2)
# plt.xlabel('Binding Energy / eV')
# plt.show()

# result.params





