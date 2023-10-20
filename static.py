import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from functions import get_tof_axis

#plots ground state spectrum

plt.style.use('style')

directory = '/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/data/'

plt.figure(figsize = (6,3))

for folder_name in ['277 Static TransDCE_N2_79Vcal','302 Static CisDCE_N2_79Vcal']:

    folder_path = '{}/{}'.format(directory,folder_name)
    tof = get_tof_axis(folder_path)

    flist = glob.glob('{}/ N=*/*'.format(folder_path))

    # for f in flist:
    #     l = os.path.basename(f)
    #     data = np.genfromtxt(f)[:,0]
    #     plt.plot(tof,data,label = l)

    data = np.genfromtxt(flist[0])[:,0]
    plt.plot(tof,data,label = folder_name)
    # np.savetxt('calib_n2.txt',np.c_[tof,data])

plt.legend()
plt.xlim(3550,3850)

plt.xlabel('etof / ns')
plt.ylabel('counts')

plt.tight_layout()
plt.show()


