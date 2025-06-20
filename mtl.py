from scipy.io import loadmat
from scripts.mtl.base import MultiTaskBase
from scripts.mtl.linear import MultiTaskLinear
import numpy as np

data = loadmat("./testdata.mat")
X = data["T_X2d"][0]
y = data["T_y"][0]


linear = MultiTaskLinear()
linear.fit_prior(X, y)
