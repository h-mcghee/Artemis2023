import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scipy
from scipy import optimize
from lmfit import Model
from lmfit import Parameters
from functions import conv, gaussian, exp
import os

plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Style_files/style')

"""load data file and set parameters"""

isomer = 'cis'
V = 79
eBE = [6.1,8.3]
eBE = [8.3,9.1]
eBE = [10.4,11.6]
eBE = [11.8,12.0]


in_file = 'processed_data/{}_{}V_delay_{}-{}eV.txt'.format(isomer,V,eBE[0],eBE[1])
root, extension = os.path.splitext(in_file)
outfile = root + "_fit.txt"

data = np.genfromtxt(in_file)

x = data[:,0]
y = data[:,1] / np.max(abs(data[:,1]))

exp1 = lambda x, t0, sigma, A1, tau1: conv(x, t0, sigma, A1, tau1)
exp2 = lambda x, t0, sigma, A1, tau1,A2,tau2: conv(x, t0, sigma, A1, tau1,A2,tau2)
exp3 = lambda x, t0, sigma, A1, tau1,A2,tau2,A3,tau3: conv(x, t0, sigma, A1, tau1,A2,tau2,A3,tau3)
exp_only = lambda x, A1, tau1: exp(x, A1, tau1)

model = Model(exp1)

params = Parameters()
params.add('t0', value=3.7,min = -50,max = 50,vary = False)
params.add('sigma', value=81,min = 0,vary = False)
params.add('A1', value = -1,max = 0 ,vary = True)
params.add('tau1', value= 91000,min = 0,vary = True)
# params.add('A2', value = 1,min = 0, vary = True)
# params.add('tau2', value=1386,min = 0, vary = True)
# params.add('A3', value=0.25, min=0,vary = True)
# params.add('tau3', value=2435, min=0,vary = True)

init_params = params.copy()

result = model.fit(y, params, x=x)
# result.params.add('fwhm',2 * np.sqrt(2 * np.log(2)) * result.params['sigma'].value)

components = True

fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1],sharex = ax1)

ax1.plot(x,y,label = 'Exp.',color = 'k')
ax1.plot(x,result.best_fit,label = 'Fit (red_chisqr = {:3f})'.format(result.redchi))
ax2.plot(x,y-result.best_fit,color = 'green')
ax2.hlines(0,-2000,10000,linestyle = '--',color = 'k')

params = np.array(result.params)
num_params = len(params) // 2
param_names = ['t0', 'sigma']

if components == True:
    gaussian_component = gaussian(x, params[0], params[1])
    ax1.fill_between(data[:, 0], gaussian_component, label=r'IRF ($\sigma$ = {:2.2f} fs)'.format(result.params['sigma'].value), linestyle='--',alpha = 0.3)
    for i in range(num_params-1):
        A_i = params[2 * i + 2]
        tau_i = params[2 * i + 3]
        # exp_i_component = exp(x, A_i, tau_i)
        exp_i_component = conv(x, params[0],params[1],A_i, tau_i)
        ax1.plot(x, exp_i_component, label=r'Exp {} ($\tau$ = {:.0f} fs)'.format(i+1,result.params['tau'+str(i+1)].value), linestyle='--')
else:
    pass

ax1.set_title(os.path.basename(in_file))
# plt.xlim(-1000,3000)

plt.xlabel('Delay / fs')
ax1.set_ylabel('Norm. intensity / arb. units')
ax2.set_ylabel('Residual')
ax1.legend()
# plt.xscale('symlog',linthresh = 1000)
plt.tight_layout()
plt.show()

# save outfile

with open(outfile, "w") as file:
    file.write("{}\n".format(os.path.basename(root)))
    file.write("\nBest Fit Data:\n")
    np.savetxt(file, np.c_[x, result.best_fit])
    # Loop through each parameter in the result.params object
    for param_name, param in result.params.items():
        file.write("\nParameter: {}\n".format(param_name))
        file.write("  Initial Value: {}\n".format(param.init_value))
        file.write("  Fitted Value: {}\n".format(param.value))
        file.write("  Standard Error: {}\n".format(param.stderr))
        file.write("  Bounds: ({}, {})\n".format(param.min, param.max))
        file.write("\n")

result.params


