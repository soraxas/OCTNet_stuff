import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
# import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math

# Environment Encoder

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(2800, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
			
	def forward(self, x):
		x = self.encoder(x)
		return x


DATASET_ROOT = "../../dataset"
DATASET_ROOT = "../../dataset_traj-occ120"

#N=number of environments; NP=Number of Paths
def load_dataset(N=100,NP=4000):

	Q = Encoder()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()

		
	obs_rep=np.zeros((N,28),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile(DATASET_ROOT+'/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)/2,2)
		obstacles=np.zeros((1,2800),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[i]=output.numpy()



	
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=DATASET_ROOT+'/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)/2,2)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=DATASET_ROOT+'/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)/2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					

	dataset=[]
	targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(32,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]
					data[28]=paths[i][j][m][0]
					data[29]=paths[i][j][m][1]
					data[30]=paths[i][j][path_lengths[i][j]-1][0]
					data[31]=paths[i][j][path_lengths[i][j]-1][1]
						
					targets.append(paths[i][j][m+1])
					dataset.append(data)
			
	data=zip(dataset,targets)
	random.shuffle(data)	
	dataset,targets=zip(*data)
	return 	np.asarray(dataset),np.asarray(targets)


# print(dataset.shape)
# print(targets.shape)
#
# asdasd

#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000):

	WTF = True
	if WTF:
		obc = None
	else:
		obc=np.zeros((N,7,2),dtype=np.float32)
		temp=np.fromfile(DATASET_ROOT+'/obs.dat')
		obs=temp.reshape(len(temp)/2,2)

		temp=np.fromfile(DATASET_ROOT+'/obs_perm2.dat',np.int32)
		perm=temp.reshape(77520,7)

		## loading obstacles
		for i in range(0,N):
			for j in range(0,7):
				for k in range(0,2):
					obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	Q = Encoder()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()


	import glob
	import regex

	###############################################################
	obc_filelist = glob.glob(DATASET_ROOT+'/obs_cloud/obc*.dat')

	# use all maps in our custom dataset
	if "traj-occ120" not in DATASET_ROOT:
		N = len(obc_filelist)
		NP = NP
	else:
		N = 10
		obc_filelist = obc_filelist[:N]
		print(len(obc_filelist))

	obc_filelist.sort(key=lambda x : int(regex.search('.*obc([0-9]+).dat', x).group(1)))
	###############################################################

	# obs_rep=np.zeros((N,28),dtype=np.float32)
	obs_rep=np.zeros((N,28),dtype=np.float32)
	k=0
	all_obstacles = []
	for fn in obc_filelist:
		temp=np.fromfile(fn)
		temp=temp.reshape(len(temp)/2,2)
		all_obstacles.append(temp)

		obstacles=np.zeros((1,2800),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k += 1

	# store all path file list
	all_path_filelist = []
	for i in range(len(obc_filelist)):
		path_filelist = glob.glob(DATASET_ROOT + '/e' + str(i) + '/path*.dat')
		path_filelist.sort(
			key=lambda x: int(regex.search('.*path([0-9]+).dat', x).group(1)))
		all_path_filelist.append(path_filelist[:NP])


	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i, path_filelist in enumerate(all_path_filelist):
		for j, fname in enumerate(path_filelist):
			path=np.fromfile(fname)
			path=path.reshape(-1,2)
			path_lengths[i][j]=len(path)
			if len(path)> max_length:
				max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i, path_filelist in enumerate(all_path_filelist):
		for j, fname in enumerate(path_filelist):
			path=np.fromfile(fname)
			path=path.reshape(-1,2)
			# path += 10
			for k in range(0,len(path)):
				paths[i][j][k]=path[k]


		##############################
		import matplotlib.pyplot as plt
		plt.scatter(*all_obstacles[i].T)
		for j in range(len(paths[i])):
			p = paths[i][j][:path_lengths[i][j]]
			plt.plot(*p.T)
		plt.show()

					
	print(path_lengths)
	return 	obc,obs_rep,paths,path_lengths, all_obstacles
	
def load_test_dataset(N=100,NP=200, s=0,sp=4000):

	WTF = True
	if WTF:
		obc = None
	else:
		obc=np.zeros((N,7,2),dtype=np.float32)
		temp=np.fromfile(DATASET_ROOT+'/obs.dat')
		obs=temp.reshape(len(temp)/2,2)

		temp=np.fromfile(DATASET_ROOT+'/obs_perm2.dat',np.int32)
		perm=temp.reshape(77520,7)

		## loading obstacles
		for i in range(0,N):
			for j in range(0,7):
				for k in range(0,2):
					obc[i][j][k]=obs[perm[i+s][j]][k]


	Q = Encoder()
	#Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	Q.load_state_dict(torch.load('AE/models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()

	obs_path_pair = load_map_paths()

	# import glob
	# import regex
	#
	# ###############################################################
	# obc_filelist = glob.glob(DATASET_ROOT+'/obs_cloud/obc*.dat')
	#
	# # use all maps in our custom dataset
	if "traj-occ120" in DATASET_ROOT:
		N = len(obs_path_pair)
		NP = NP
	else:
		N = 100
		# obc_filelist = obc_filelist[:N]
		# print(len(obc_filelist))
	#
	# obc_filelist.sort(key=lambda x : int(regex.search('.*obc([0-9]+).dat', x).group(1)))
	###############################################################

	# print(obs_path_pair)
	import matplotlib.pyplot as plt
	#
	# for obs, paths in obs_path_pair:
	# 	plt.scatter(*obs.reshape(-1,2).T)
	# 	for p in paths:
	# 		plt.plot(*p.reshape(-1,2).T)
	# 	plt.show()



	# obs_rep=np.zeros((N,28),dtype=np.float32)
	obs_rep=np.zeros((N,28),dtype=np.float32)
	k=0
	all_obstacles = []
	for obs, _ in obs_path_pair:
		obs=obs.reshape(-1,2)
		all_obstacles.append(obs)

		inp=torch.from_numpy(obs.copy().flatten())
		inp=Variable(inp).cuda()
		output=Q(inp.float())
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k += 1


	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i, (_, paths) in enumerate(obs_path_pair):
		for j, path in enumerate(paths[:NP]):
			path=path.reshape(-1,2)
			path_lengths[i][j]=len(path)
			if len(path)> max_length:
				max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i, (_, _paths) in enumerate(obs_path_pair):
		for j, path in enumerate(_paths[:NP]):
			path=path.reshape(-1,2)
			# path += 10
			paths[i][j][:len(path)] = path
			# for k in range(0,len(path)):


		##############################
		# import matplotlib.pyplot as plt
		# plt.scatter(*all_obstacles[i].T)
		# for j in range(len(paths[i])):
		# 	p = paths[i][j][:path_lengths[i][j]]
		# 	plt.plot(*p.T)
		# plt.show()


	print(path_lengths)
	return 	obc,obs_rep,paths,path_lengths, all_obstacles



def load_map_paths():
	import matplotlib.pyplot as plt
	import glob

	obs_path_pair = []

	num_obs = len(glob.glob(DATASET_ROOT + '/obs_cloud/obc*.dat'))

	for i in range(num_obs):
		fname = DATASET_ROOT + '/obs_cloud/obc{}.dat'.format(i)
		obs = np.fromfile(fname)

		# plt.scatter(*obs.reshape(-1,2).T)
		num_path = len(glob.glob('{}/e{}/path{}.dat'.format(DATASET_ROOT, i, '*')))
		paths = []
		for j in range(num_path):
			path = np.fromfile('{}/e{}/path{}.dat'.format(DATASET_ROOT, i, j))
			paths.append(path)
		# 	plt.plot(*path.reshape(-1,2).T)
		# plt.show()
		obs_path_pair.append((obs, paths))

	import random
	random.shuffle(obs_path_pair)
	return obs_path_pair

# load_map_paths()
