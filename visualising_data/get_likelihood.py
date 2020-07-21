import sklearn
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

clustering = sklearn.cluster.DBSCAN(eps=5)


visualise = True
visualise = False



#%%
# calculate likelihood

def get_likelihood(gt, datas):
    stdevs = np.array(gt).std(axis=0)
    means = np.array(gt).mean(axis=0)

    ll = np.sum(np.log(2*np.pi*(stdevs**2))/2 + ((datas-means)**2)/(2 * (stdevs**2)), axis=0)
    return ll

    # likelihood = (2*np.pi*stdevs**2)**(-len(gt)/2) * np.exp(-((datas-means)**2).sum(axis=0)/(2*stdevs**2))
    # # likelihood = (-len(gt)/2) * np.exp(-((datas-means)**2).sum(axis=0)/(2*stdevs**2))
    # return likelihood

# xs


#%%
import scipy
import scipy.spatial.distance
import tqdm


testdata = np.load("test_data.npy", encoding='bytes', allow_pickle=True)
traindata = np.load("train_data.npy", encoding='bytes', allow_pickle=True)

def get_pts(obs):
    pts = []
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i, j]:
                pts.append((i, j))
    return pts


closest_idx = []
for obstest, _ in tqdm.tqdm(testdata):
    obstest = get_pts(obstest)
    dists = []
    for obstrain, _ in traindata:
        obstrain = get_pts(obstrain)
        dists.append(scipy.spatial.distance.directed_hausdorff(obstest, obstrain)[0])
    # break
    closest_idx.append(np.array(dists).argmin())

#%%


def normalise_steps(arr, norm_to_steps=100):
    norm_steps = np.linspace(1, 100, norm_to_steps)
    _ori_steps = np.linspace(1, 100, len(arr))
    return np.interp(norm_steps, _ori_steps, arr)

likelihood_over_maps = []
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

    traj_class = [[] for _ in range(len(separated_paths))]

    test_xss_mean = np.array([np.array(_p).mean(axis=0) for _p in xss])
    test_yss_mean = np.array([np.array(_p).mean(axis=0) for _p in yss])

    test_norm_paths = np.stack([test_xss_mean, test_yss_mean])
    test_xss_mean.shape
    test_yss_mean.shape
    test_norm_paths.shape

    train_path_class = []

    for p in traindata[closest_idx[test_idx], 1]:
        _train_xs = normalise_steps(p[:,0],100)
        _train_ys = normalise_steps(p[:,1],100)
        train_norm_path = np.stack([_train_xs, _train_ys])

        train_norm_path.shape
        test_norm_paths.shape
        # _test.shape

        closest_class = []
        for i in range(test_norm_paths.shape[1]):
            _test = test_norm_paths[:,i,:]
            closest_class.append(np.linalg.norm(train_norm_path - _test))
        _idx = np.array(np.array(closest_class).argmin())

        train_path_class.append(_idx)









    for i, (xs, ys) in enumerate(zip(xss, yss)):


        train_paths = traindata[closest_idx[test_idx], 1]
        train_paths_of_this_class = [train_paths[_] for _ in range(len(train_paths)) if train_path_class[_] == i]
        train_paths_of_this_class_x = [normalise_steps(p[0], 100) for p in train_paths_of_this_class]
        train_paths_of_this_class_y = [normalise_steps(p[0], 100) for p in train_paths_of_this_class]


        get_likelihood(gt=np.array(xs), datas=np.array(xs))
        x_likelihood = get_likelihood(gt=np.array(xs), datas=np.array(train_paths_of_this_class_x))
        y_likelihood = get_likelihood(gt=np.array(ys), datas=np.array(train_paths_of_this_class_y))

        likelihood_over_classes.append((x_likelihood + y_likelihood).mean())

    likelihood_over_maps.append(max(likelihood_over_classes))
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



np.array(likelihood_over_maps).mean()
