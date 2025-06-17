from scipy.io import loadmat
from scripts.mtl.base import MultiTaskBase
import numpy as np

data = loadmat("./testdata.mat")
data_2d = data["T_X2d"]
labels_2d = data["T_y"][0]

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
labels = np.array([[0, 1], [1, -1]])

y = MultiTaskBase.swap_labels(y, labels, direction="to")
print(y)
