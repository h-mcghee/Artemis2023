import numpy as np
import matplotlib.pyplot as plt

plt.style.use('style')

trans = np.genfromtxt('processed_data/trans_gs.txt')
cis = np.genfromtxt('processed_data/cis_gs.txt')

plt.figure(figsize = (3,3))
plt.plot(trans[:,0],trans[:,1],label = 'trans-DCE',color = 'black')
plt.plot(cis[:,0],cis[:,1],label = 'cis-DCE',color = 'red')
plt.xlim(7,21)
plt.legend()
plt.xlabel('Binding Energy / eV')
plt.ylabel('Normalised Intensity / arb. units')
plt.tight_layout()
# plt.savefig('gs.png',dpi = 500,bbox_inches='tight')

plt.show()
