import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *

plt.style.use('style')

jacobian = False
save = False
bkg_sub = True

vlineout = [3.0,8.3]
vlineout = [8.3,9.1]
vlineout = [10.4,11.6]
vlineout = [11.8,12.0]

# tof,df,fname = process_energy(275)
# tof2,df2,fname2 = process_energy(289)

# # combine datasets - could maybe write a function for this?
# total = pd.concat([df,df2])
# total.delay = total.delay.apply(lambda x: x[0]) # takes values out of square brackets - might need to change from functions
# df = total.groupby('delay',as_index = False).sum()

tof,df,fname = process_energy(305)

#load corresponding calibration file
try:
    params = np.loadtxt('calibrations/{}_calib.txt'.format(fname))
except:
    raise Exception('No calibration file found for {}'.format(fname))

x = tof2energy(tof,*params)
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


bkg = tof_slice(y,-1500,-300)
gs = np.mean(z[bkg,:],axis = 0)
gs = gs/np.max(gs)
bkg_matrix = np.mean(z[bkg,:],axis = 0)[None,:]


if bkg_sub:
    z = z - bkg_matrix
    
else:
    pass



#heatmap

plt.figure(figsize = (3,3))
plt.pcolormesh(y,x,z.T,vmin = 0,vmax = 50)
plt.colorbar()
# plt.xlim(-100,1000)
plt.ylim(3,9.5)
plt.xlabel('Delay / fs')
plt.ylabel('eBE / eV')
plt.gca().invert_yaxis()
plt.show()

#Kinetic slice

plt.figure(figsize = (6,3))


hlineout = [20,40]

h = np.sum(z[:,tof_slice(x,vlineout[0],vlineout[1])],axis=1) 
h_error = np.sqrt(h)
bkg = np.sum(bkg_matrix[:,tof_slice(x,vlineout[0],vlineout[1])],axis=1)
bkg_error = np.sqrt(bkg)
error = np.sqrt(h_error**2 + bkg_error**2)


# plt.plot(y,h,label = '{:.1f}-{:.1f} eV'.format(vlineout[0],vlineout[1]))
plt.errorbar(y,h-bkg,error,label = '{:.1f}-{:.1f} eV'.format(vlineout[0],vlineout[1]))
plt.xlabel('Delay / fs')
plt.ylabel('Counts')
plt.legend()
# plt.xscale('symlog',linthreshx = 1000)
if save:
    np.savetxt('processed_data_v2/cis_79V_delay_{:.1f}-{:.1f}eV.txt'.format(vlineout[0],vlineout[1]),np.c_[y,h-bkg,error])
plt.show()

plt.figure(figsize = (6,3))
plt.plot(x,np.mean(z[tof_slice(y,hlineout[0],hlineout[1]),:],axis=0),label = '{}-{} fs'.format(hlineout[0],hlineout[1]))
plt.plot(x,gs*1000,label = 'gs')
plt.xlabel('eBE / eV')
plt.ylabel('Counts')
plt.xlim(3,10)
# plt.ylim(0,50)
plt.legend()
plt.show()


plt.figure(figsize = (6,3))
for i in [[-100,50],[50,100],[100,200],[200,300],[300,400],[400,500]]:
    plt.plot(x,np.mean(z[tof_slice(y,i[0],i[1]),:],axis=0),label = '{}-{} fs'.format(i[0],i[1]))
plt.xlabel('eBE / eV')
plt.ylabel('Counts')
plt.legend()
plt.xlim(3,10)
plt.ylim(0,30)
plt.show()

plt.figure(figsize = (6,3))
for i in [[2.5,6.1],[6.1,8.3]]:
    plt.plot(y,np.sum(z[:,tof_slice(x,i[0],i[1])],axis=1)/np.max(np.sum(z[:,tof_slice(x,i[0],i[1])],axis=1)),label = '{}-{} eBE'.format(i[0],i[1]))
plt.xlabel('Delay / fs')
plt.ylabel('Counts')
plt.xscale('symlog',linthreshx = 1500)
plt.legend()
plt.show()


