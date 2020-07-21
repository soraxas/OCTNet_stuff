
import numpy as np
import matplotlib.pyplot as plt



# cd ~/research/MPNet/dataset

# ls

i = 10

map = np.fromfile(f'OccTraj_in_mpnet_format/obs_cloud/obc{i}.dat').reshape(-1,2)


#%%

plt.scatter(*map.T)

for j in range(100):
    path = np.fromfile(f'OccTraj_in_mpnet_format/e{i}/path{j}.dat').reshape(-1,2)
    plt.plot(*path.T)

plt.show()
