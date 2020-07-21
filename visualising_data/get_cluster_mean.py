#%%
import sklearn
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
import glob

clustering = sklearn.cluster.DBSCAN(eps=5)


visualise = True
visualise = False

max_path_length = 0

clustered_starts_ends = []
SAVE_BLOB = []

for map_idx in range(31 + 1):

    # ls OccTraj_in_mpnet_format/obs_cloud

    obs = np.fromfile(f"OccTraj_in_mpnet_format/obs_cloud/obc{map_idx}.dat")

    if visualise:
        plt.scatter(*obs.reshape(-1,2).T)

    starts = []
    ends = []

    paths = []

    for fn in glob.glob(f"OccTraj_in_mpnet_format/e{map_idx}/*.dat"):
        path = np.fromfile(fn).reshape(-1,2)
        max_path_length = max(max_path_length, len(path))
        if visualise:
            plt.plot(*path.T, linewidth=.12)

        starts.append(path[0])
        ends.append(path[-1])
        paths.append(path)

        # break

    starts =  np.array(starts)
    ends =  np.array(ends)

    clustered_starts_end = ([], [])

    for pts, color in [(starts, 'red'), (ends, 'blue')]:
        labels = clustering.fit_predict(pts)


        for l in np.unique(labels):
            mean_pt = np.array(pts)[labels == l].mean(axis=0)
            if visualise:
                plt.scatter(*mean_pt, marker='o', color=color, s=50)

            _idx = 0 if color == 'red' else 1
            clustered_starts_end[_idx].append(mean_pt)

    clustered_starts_ends.append(clustered_starts_end)


    # print([len(p) for p in clustered_starts_ends])
    # break

    # plt.scatter(*np.array(ends)[clustering.fit_predict(ends)].mean(axis=0), marker='o', color='blue', s=50)
    if visualise:
        plt.show()
        if map_idx > 2:
            break

    SAVE_BLOB.append(obs)
#%%

clustered_starts_ends = []
SAVE_BLOB = []

for map_idx in range(31 + 1):
    obs = np.fromfile(f"OccTraj_in_mpnet_format/obs_cloud/obc{map_idx}.dat")
# for obs in SAVE_BLOB:
#     obs = np.fromfile(f"OccTraj_in_mpnet_format/obs_cloud/obc{map_idx}.dat")

    if visualise:
        plt.scatter(*obs.reshape(-1,2).T)

    starts = []
    ends = []

    paths = []

    for fn in glob.glob(f"OccTraj_in_mpnet_format/e{map_idx}/*.dat"):
        path = np.fromfile(fn).reshape(-1,2)
        if visualise:
            plt.plot(*path.T, linewidth=.12)

        starts.append(path[0])
        ends.append(path[-1])
        paths.append(path)

        # break

    starts =  np.array(starts)
    ends =  np.array(ends)

    clustered_starts_end = ([], [])

    # for pts, color in [(starts, 'red'), (ends, 'blue')]:
    for pts, color in [(ends, 'blue')]:
        labels = clustering.fit_predict(pts)



        for _i, l in enumerate(labels):
            colours = ['red', 'blue', 'green', 'purple']
            plt.plot(*paths[_i].T, c=colours[l])



    clustered_starts_ends.append(clustered_starts_end)


    # print([len(p) for p in clustered_starts_ends])
    # break

    # plt.scatter(*np.array(ends)[clustering.fit_predict(ends)].mean(axis=0), marker='o', color='blue', s=50)

    # plt.show()
    # if map_idx > 2:
    #     break

    SAVE_BLOB.append(obs)





#%%


def normalise_steps(arr, norm_to_steps=100):
    norm_steps = np.linspace(1, 100, norm_to_steps)
    _ori_steps = np.linspace(1, 100, len(arr))
    return np.interp(norm_steps, _ori_steps, arr)


clustered_starts_ends = []
SAVE_BLOB = []
visualise = True
visualise = False
for map_idx in range(31 + 1):
    obs = np.fromfile(f"OccTraj_in_mpnet_format/obs_cloud/obc{map_idx}.dat")
# for obs in SAVE_BLOB:

    if visualise:
        plt.scatter(*obs.reshape(-1,2).T)

    starts = []
    ends = []

    paths = []

    for fn in glob.glob(f"OccTraj_in_mpnet_format/e{map_idx}/*.dat"):
        path = np.fromfile(fn).reshape(-1,2)
        starts.append(path[0])
        ends.append(path[-1])
        paths.append(path)

    # plt.plot(np.array(xs).mean(axis=0), np.array(ys).mean(axis=0), linewidth=3, c='red')

        # break

    starts =  np.array(starts)
    ends =  np.array(ends)

    # find traj classes
    pts = ends

    labels = clustering.fit_predict(pts)

    separated_paths = [[p for (_i, p) in enumerate(paths) if labels[_i] == l] for l in np.unique(labels)]

    # [a[0] for a in separated_paths]

    xss = []
    yss = []
    for _class, _paths in enumerate(separated_paths):
        colours = ['red', 'blue', 'green', 'purple']
        colours2 = ['lime', 'magenta', 'gold', 'grey']

        _paths

        xs = []
        ys = []
        for _p in _paths:
            if visualise:
                plt.plot(*_p.T, c=colours[l], linewidth=.5)

            xs.append(normalise_steps(_p[:,0], 100))
            ys.append(normalise_steps(_p[:,1], 100))

        xss.append(xs)
        yss.append(ys)

    for i, (xs, ys) in enumerate(zip(xss, yss)):
        if visualise:
            plt.plot(np.array(xs).mean(axis=0), np.array(ys).mean(axis=0), linewidth=5, c=colours2[i])


    clustered_starts_ends.append(clustered_starts_end)



    if visualise:
        plt.show()
    # break
    # continue
    # if map_idx > 2:
    #     break

    # SAVE_BLOB.append(obs)





#%%
# calculate likelihood

datas = np.array(xs)
stdevs = np.array(xs).std(axis=0)
means = np.array(xs).mean(axis=0)

likelihood = (2*np.pi*stdevs**2)**(-len(means)/2) * np.exp(-((datas-means)**2).sum(axis=0)/(2*stdevs**2))


def get_likelihood(gt, datas):
    stdevs = np.array(gt).std(axis=0)
    means = np.array(gt).mean(axis=0)

    likelihood = (2*np.pi*stdevs**2)**(-len(gt)/2) * np.exp(-((datas-means)**2).sum(axis=0)/(2*stdevs**2))
    # likelihood = (-len(gt)/2) * np.exp(-((datas-means)**2).sum(axis=0)/(2*stdevs**2))
    return likelihood

plt.plot(get_likelihood(xs, xs))
np.array(xs).mean(axis=0)
np.array(xs).std(axis=0)


#%%

import pprint
clustered_starts_ends

cluster_start_end_pairs = []

pprint.pprint([[len(_p) for _p in p] for p in clustered_starts_ends])

for start_ends in clustered_starts_ends:
    cluster_start_end_pairs.append([])
    for start in start_ends[0]:
        for end in start_ends[1]:
            if np.linalg.norm(start - end) < 3:
                # suppress start, end being very close
                continue
            cluster_start_end_pairs[-1].append((start, end))




pprint.pprint([len(p) for p in cluster_start_end_pairs])
pprint.pprint([[list(tuple(_) for _ in _p) for _p in p] for p in cluster_start_end_pairs])
pprint.pprint([[list(map(tuple, pair)) for pair in pairs] for pairs in cluster_start_end_pairs])



#%%

np.save('block.npy', (
     SAVE_BLOB,
     [[list(map(tuple, pair)) for pair in pairs] for pairs in cluster_start_end_pairs],
))

#%%
import scipy
import scipy.spatial.distance
import tqdm

# scipy.spatial.distance.directed_hausdorff()

testdata = np.load("test_data.npy", encoding='bytes')
traindata = np.load("train_data.npy", encoding='bytes')

def get_pts(obs):
    pts = []
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i, j]:
                pts.append((i, j))
    return pts

get_pts(obstest)

closest_idx = []
for obstest, _ in tqdm.tqdm(testdata):
    obstest = get_pts(obstest)
    dists = []
    for obstrain, _ in traindata:
        obstrain = get_pts(obstrain)
        dists.append(scipy.spatial.distance.directed_hausdorff(obstest, obstrain)[0])
    # break
    closest_idx.append(np.array(dists).argmin())



def normalise_steps(arr, norm_to_steps=100):
    norm_steps = np.linspace(1, 100, norm_to_steps)
    _ori_steps = np.linspace(1, 100, len(arr))
    return np.interp(norm_steps, _ori_steps, arr)

paths = testdata[0][1]
for test_idx, (obs, paths) in enumerate(testdata):
    starts, ends = [], []
    for p in paths:
        starts.append(p[0])
        ends.append(p[-1])

    starts =  np.array(starts)
    ends =  np.array(ends)

    # find traj classes
    labels = clustering.fit_predict(ends)


    separated_paths = [[p for (_i, p) in enumerate(paths) if labels[_i] == l] for l in np.unique(labels)]
    xss = []
    yss = []
    likelihood_over_classes = []
    for _class, _paths in enumerate(separated_paths):
        colours = ['red', 'blue', 'green', 'purple']
        colours2 = ['lime', 'magenta', 'gold', 'grey']

        _paths

        xs = []
        ys = []
        for _p in _paths:
            if visualise:
                plt.plot(*_p.T, c='red', linewidth=.5)

            xs.append(normalise_steps(_p[:,0], 100))
            ys.append(normalise_steps(_p[:,1], 100))

        xss.append(xs)
        yss.append(ys)

    for i, (xs, ys) in enumerate(zip(xss, yss)):


        closest_train_paths = traindata[closest_idx[test_idx], 1]

        train_paths_x = [normalise_steps(closest_train_paths[__][:,0], 100) for __ in range(len(closest_train_paths))]
        train_paths_y = [normalise_steps(closest_train_paths[__][:,1], 100) for __ in range(len(closest_train_paths))]


        x_likelihood = get_likelihood(gt=np.array(xs), datas=np.array(train_paths_x))
        y_likelihood = get_likelihood(gt=np.array(ys), datas=np.array(train_paths_y))

        likelihood_over_classes.append((x_likelihood + y_likelihood).mean())

    likelihood_over_classes
    #     plt.plot(get_likelihood(gt=np.array(xs), datas=np.array(train_paths_x)), label=f"{i}x")
    #     plt.plot(get_likelihood(gt=np.array(ys), datas=np.array(train_paths_y)), label=f"{i}y")
    #     # get_likelihood(gt=np.array(xs), datas=np.array(train_paths_x))
    # plt.legend()
    # plt.show()




    if visualise:
        plt.show()
    # break
    # continue
    # if map_idx > 2:
    #     break

    # SAVE_BLOB.append(obs)


# %%
