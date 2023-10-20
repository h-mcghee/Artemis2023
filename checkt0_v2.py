import numpy as np 
import matplotlib.pyplot as plt
import os

from functions import get_fpath
from functions import make_runmap
from functions import get_tof_axis
from functions import get_delay_time
from functions import tof_slice
import glob

try:
    plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Style_files/style')
except:
    pass
    print("*style file not found* - use style file for pretty plots")

# write function which retrieves array of all the N files

data_path = '/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/data'


def get_N_files(folder_path):

    "returns a list of the N files in a given folder"

    flist = os.listdir(folder_path)
    directories = [int(item[3:]) for item in flist if os.path.isdir(os.path.join(folder_path, item))]
    return sorted(directories)

def process_delay(run,tof_window,N):

    "Same function as in delay.py but pulling N out as an argument for the function"
    df = make_runmap('data/Data Log.tsv')
    t0 = df[df.run==run].t0.values

    fpath = get_fpath(data_path,str(run))[0]
    fname = os.path.basename(fpath)

    delay = []
    sum_counts = []

    flist = sorted(glob.glob('{}/ N={}/*'.format(fpath,int(N))))

    tof = get_tof_axis(fpath)

    ROI = tof_slice(tof,tof_window[0],tof_window[1])

    for f in flist:
        data = np.genfromtxt(f)[:,0]
        delay.append(get_delay_time(f,t0))
        sum_counts.append(np.sum(data[ROI]))

    return delay, sum_counts, fname

plt.figure(figsize = (6,3))
run = 275
folder_path = get_fpath(data_path,str(run))[0]
N_files = get_N_files(folder_path)
tof_window = [3560,3601]
# tof_window = [3610,3619]



for (f1,f2) in zip(N_files,N_files[1:]):
    delay,lb,fname = process_delay(run,tof_window,f1)
    ub = process_delay(run,tof_window,f2)[1]
    diff = np.array(ub)-np.array(lb)

    plt.plot(delay,diff/np.max(diff),label = '{}:{} cycles'.format(f1,f2))

plt.legend()
plt.title(fname)
plt.xlim(-500,500)
plt.xlabel('Delay / fs')
plt.show()



