import matplotlib.pyplot as plt
import numpy as np

out = np.load("prediction_outputs.npy", allow_pickle=True, encoding='bytes')

# elements in out = [....]
# dict_keys(['pred-neural', 'map', 'gt', 'pred-rrt'])


for obj in out:
    plt.scatter(*obj['map'].reshape(-1, 2).T)
    for i in range(len(obj['gt'])):
        _gt                      = obj['gt'][i]
        _rrt_fesible, _rrt       = obj['pred-rrt'][i]
        _neural_fesible, _neural = obj['pred-neural'][i]
        plt.scatter(*_gt[0], color='green', marker='o', s=100)
        plt.scatter(*_gt[1], color='lime', marker='^', s=100)
        if _rrt is not None:
            plt.plot(*_rrt.T, color='red')
        if _neural is not None:
            plt.plot(*_neural.T, color='blue')

    plt.show()
    # break
