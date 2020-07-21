# import torch
# import torch.utils.data as data
# import os
# import pickle
import numpy as np
# import nltk
# from PIL import Image
# import os.path
# import random

import glob

def load_dataset(N=30000,NP=1800):

	obstacles=np.zeros((N,2800),dtype=np.float32)
	for i, fn in enumerate(glob.glob('../../../dataset_traj-occ120/obs_cloud/obc*.dat')):
		temp = np.fromfile(fn)
		temp=temp.reshape(len(temp)/2,2)
		obstacles[i]=temp.flatten()
	# for i in range(0,N):
	# 	temp=np.fromfile('../dataset2/obs_cloud/obc'+str(i)+'.dat')
	# 	temp=temp.reshape(len(temp)/2,2)
	# 	obstacles[i]=temp.flatten()

	
	return 	obstacles	
