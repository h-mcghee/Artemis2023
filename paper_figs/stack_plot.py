import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.style.use('/Users/harrymcghee/Dropbox (UCL)/2_HMG/Artemis2023/style')

isomer = 'cis'
voltage = '79'

colors = plt.cm.viridis(np.linspace(0, 0.6, 5))

ranges = [(6.1,8.3),(8.3,9.1),(10.4,11.6),(11.8,12.0)]

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import numpy as np

def read_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        if line.strip() == "Best Fit Data:":
            start_line = i + 1
        elif start_line is not None and not line.strip():
            end_line = i
            break

    if start_line is None or end_line is None:
        raise ValueError("Could not find 'Best Fit Data' block in the file.")

    best_fit_data_lines = lines[start_line:end_line]
    best_fit_data = np.array([list(map(float, line.strip().split())) for line in best_fit_data_lines])

    return best_fit_data


plt.figure(figsize=(3,5))
offset = 1.5
for i, eBE in enumerate(ranges):
    data = np.genfromtxt(os.path.join(parent_dir,'processed_data/{}_{}V_delay_{}-{}eV.txt'.format(isomer,voltage,eBE[0],eBE[1])))
    fit_file = os.path.join(parent_dir,'processed_data/{}_{}V_delay_{}-{}eV_fit.txt'.format(isomer,voltage,eBE[0],eBE[1]))
    fit = read_file(fit_file)
    fit = fit[2:]
    x = data[2:,0]
    if i == 3:
        offset = 1.8
    normy = data[2:,1]/np.max(abs(data[2:,1]))

    ###

    plt.plot(x,i*offset + normy,'o',markersize = '2',color = colors[i])
    plt.plot(x,i*offset + fit[:,1],color = colors[i])
    plt.text(500, i*offset +0.3 , '{:.1f}-{:.1f} eV'.format(eBE[0],eBE[1]), va='top', ha='left',color = colors[i])



# plt.xscale('symlog',linthreshx = 1000)
# plt.xticks([-500,0,500,1000,1500,2000,2500,3000,3500,4000,4500],[])
plt.yticks([])
# plt.gca().xaxis.set_major_formatter(plt.NullFormatter())
plt.xlabel('Delay / fs')
plt.ylabel('Intensity / arb. units')
# plt.vlines(1000,0,5.5,linestyle = 'dashed',color = 'gray',alpha = 0.5)
# plt.xticks([])
# xticks = [0,500,1000,1500]
# xticklabels = ['{}'.format(xtick) for xtick in xticks]
# plt.xticks(xticks)
plt.xlim(-500,1000)
plt.vlines(0,-0.2,6,linestyle = 'dashed',color = 'gray',alpha = 0.5)
plt.legend()
plt.tight_layout()
# plt.savefig('cis_delay_zoom.png',dpi = 500,bbox_inches='tight')
plt.show()


######