import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from calib_functions import *
import random
from tqdm import tqdm

plt.style.use('style')

data_folder = 'processed_data_v3'

run = 275

save = False
bkg_sub = True

eBE = [3.0,8.3]

time = [500,2000]

dir_name = '{}/Run_{}'.format(data_folder,run)

#heatmap

plt.figure()
data = np.genfromtxt(os.path.join(dir_name,'main.txt'))
x = data[0,1:]
y = data[1:,0]
z = data[1:,1:]


bkg = tof_slice(y,-1500,-300)
bkg_matrix = np.mean(z[bkg,:],axis = 0)[None,:]

if bkg_sub:
    z = z - bkg_matrix 
else:
    pass

plt.pcolormesh(x,y,z)
plt.colorbar()

# delay plots with error 

# these aren't changing with jacobian

def delay_plot(file,eBE,time):
    data = np.genfromtxt(file)
    x = data[0,1:]
    y = data[1:,0]
    z = data[1:,1:]
    bkg = tof_slice(y,-1500,-300)
    bkg_matrix = np.mean(z[bkg,:],axis = 0)[None,:]
    z = z-bkg_matrix
    h = np.sum(z[:,tof_slice(x,eBE[0],eBE[1])],axis=1) 
    t = np.sum(z[tof_slice(y,time[0],time[1]),:],axis=0)
    return h,t



delay_list = []
energy_list = []
flist = [f for f in os.listdir(dir_name) if f.startswith('submatrix')]

for f in flist:
    h,t = delay_plot(os.path.join(dir_name,f),eBE,time)
    delay_list.append(h)
    energy_list.append(t)

delay_error = np.std(delay_list,axis = 0)
energy_error = np.std(energy_list,axis = 0)
h,t = delay_plot(os.path.join(dir_name,'main.txt'),eBE,time)

plt.figure()
plt.errorbar(y,h,delay_error)
plt.xlim(-1000,2000)

plt.figure()
plt.errorbar(x,t,energy_error)
# plt.xlim(2,10)
# plt.ylim(-10,200)

#multiple plots

plt.figure(figsize = (6,3))
for time in [[-200,200],[500,1000]]:
    delay_list = []
    energy_list = []
    flist = [f for f in os.listdir(dir_name) if f.startswith('submatrix')]
    for f in flist:
        h,t = delay_plot(os.path.join(dir_name,f),eBE,time)
        energy_list.append(t)
    energy_error = np.std(energy_list,axis = 0)
    h,t = delay_plot(os.path.join(dir_name,'main.txt'),eBE,time)
    plt.errorbar(x,t,energy_error,label = '{}-{} fs'.format(time[0],time[1]))
plt.ylim(-10,200)
plt.xlim(2,10)
plt.legend()
plt.xlabel('Energy / eV')
plt.ylabel('Counts')

plt.figure(figsize = (6,3))
for eBE in [[10.4,11.6]]:
    delay_list = []
    energy_list = []
    flist = [f for f in os.listdir(dir_name) if f.startswith('submatrix')]
    for f in flist:
        h,t = delay_plot(os.path.join(dir_name,f),eBE,time)
        delay_list.append(h)
    delay_error = np.std(delay_list,axis = 0)
    h,t = delay_plot(os.path.join(dir_name,'main.txt'),eBE,time)
    plt.errorbar(y,h,delay_error,label = '{}-{} eV'.format(eBE[0],eBE[1]))
# plt.ylim(-10,200)
# plt.xlim(2,10)
plt.legend()
plt.xlabel('Delay / fs')
plt.ylabel('Counts')
# plt.xlim(-500,1000)
plt.xscale('symlog',linthreshx = 1000)


# need to add save
# re-run with jacobian
# normalisation?

