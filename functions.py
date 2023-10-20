import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import scipy 
import re

data_path = '/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/data'

def get_tof_axis(folder_path):
    """
    pixel to TOF conversion

    Args:
        folder_path (str): Path to the run folder containing the TDC Time.tsv file.
    Returns:
        tof_axis (np.array): TOF axis in ns.

    """
    fpath = '{}/TDC Time.tsv'.format(folder_path)
    tof_axis = np.genfromtxt(fpath)
    return tof_axis

def get_delay_pos_list(folder_path,t0):
    """
    Gets a list of all the delay positions in mm in a given run folder
    and converts them into delay times in fs. It reads these delay positions from the file name.

    Args:
        folder_path (str): Path to the run folder.
        t0 (float): t0 value for the run

    Returns:
        delay_pos (np.array): Array of delay times in fs.
    """
    flist = sorted(glob.glob('{}/*/*'.format(folder_path)))
    delay_time = []
    for f in flist:
        delay_pos = float(os.path.basename(f)[:-4])
        delay_time.append(2*(delay_pos - t0)/0.000299792458)
    return np.unique(delay_time)

def get_delay_time(f,t0):
    """
    Converts delay position in mm for a given file into delay time in fs.
    Similar to get_delay_pos_list but for a single file.

    Args:
        f (str): Path to the file.
        t0 (float): t0 value for the run
    
    Returns:
        delay_time (float): Delay time in fs.
    """
    delay_pos = float(os.path.basename(f)[:-4])
    delay_time = (2*(delay_pos - t0)/0.000299792458)
    return delay_time 

def tof_slice(tof, start, end):
    """ 
    return a slice object that slices tof between start and end values 
    
    Args:
        tof (np.array): TOF axis in ns.
        start (float): Start of the slice in ns.
        end (float): End of the slice in ns.
    
    Returns:
        slice: A slice object that slices tof between start and end values.
    """
    start_idx = np.argmin( np.abs( tof - start ))
    stop_idx  = np.argmin( np.abs( tof - end ))
    return slice(start_idx, stop_idx)


def get_delay_data(folder_path,N,t0):
    flist = (glob.glob('{}/ N={}/*'.format(folder_path,int(N))))
    delay = []
    pes = []
    for f in flist:
        delay.append(get_delay_time(f,t0))
        pes.append(np.genfromtxt(f)[:,0])

    df = pd.DataFrame({'delay':delay,
                        'pes':pes})
    if len(delay) != 0:
        df = df.sort_values(by='delay').reset_index(drop = True)
    else:
        pass

    return df

def make_runmap(f):
    """
    takes the Data Log.tsv file from the experiment and creates a runmap dataframe
    
    Args:
        f (str): Path to the Data Log.tsv file.
    Returns:
        df (pd.DataFrame): Dataframe containing the runmap.
    """
    df = pd.read_csv(f, sep='\t', header=None)
    df.fillna(value=np.nan, inplace=True)
    headers = ['date','time','run','notes','time step','t0','sweeps','8','9','10','11','12','13','14']  
    df.columns = headers 
    return df

def get_info(run):
    df = make_runmap('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/data/Data Log.tsv')
    return df[df.run==str(run)]

def get_fpath(folder_path, search_strings):
    """
    Search for files within the given folder that contain the exact group of search strings in their names.

    Args:
        folder_path (str): The path of the folder where the search will be performed.
        search_strings (list): A list of strings representing the exact group to be found in the file names.

    Returns:
        list: A list of paths of all the matching files found. An empty list if no matching file is found.
    """
    group_search_string = ''.join(search_strings)

    matching_files = []

    for filename in os.listdir(folder_path):
        if re.search(r'\b' + re.escape(group_search_string) + r'\b', filename):
            matching_files.append(os.path.join(folder_path, filename))

    return matching_files

def process_delay(run,tof_window):
    df = make_runmap('data/Data Log.tsv')
    t0 = df[df.run==run].t0.values
    N = df[df.run==run].sweeps.values -1

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

def process_energy(run):
    df = make_runmap('data/Data Log.tsv')
    t0 = df[df.run==run].t0.values
    N = df[df.run==run].sweeps.values -1

    fpath = get_fpath(data_path,str(run))[0]
    fname = os.path.basename(fpath)

    tof = get_tof_axis(fpath)

    df = get_delay_data(fpath,N,t0)

    return tof,df,fname

def conv(x,t0,sigma, *args):
    num_exponentials = len(args) // 2
    num_params = len(args)

    if num_params % 2 != 0 or num_exponentials == 0:
        raise ValueError("The number of parameters must be a multiple of 2 (A, tau) for each exponential.")

    # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    result = np.zeros_like(x)

    for i in range(num_exponentials):
        A = args[2*i]
        tau = args[2*i + 1]
        exp_term = A / 2 * np.exp((-1 / tau) * (x - t0)) * np.exp((sigma**2) / (2 * tau**2)) * \
           (1 + scipy.special.erf((x - t0 - ((sigma **2) / tau)) / (np.sqrt(2.0) * sigma)))
        result += exp_term

    return result

def gaussian(x, t0, sigma):
    # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-((x - t0) ** 2) / (2 * sigma ** 2))

def exp(x,A,tau):
    return A*np.exp(-x/tau)

def idx(x, value):
    return np.argmin(np.abs(x - value))




