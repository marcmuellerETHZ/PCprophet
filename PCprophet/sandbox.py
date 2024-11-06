import re
import sys
import os
import numpy as np
import scipy.signal as signal
import pandas as pd
from scipy.ndimage import uniform_filter
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

import PCprophet.parse_go as go
import PCprophet.io_ as io
import PCprophet.stats_ as st


data = pd.read_csv("test/test_frac.txt")


a, b = pairs[0].get_inte(), pairs[1].get_inte()
# a,b are input arrays; W is window length

#MM: smoothing
am = uniform_filter(a.astype(float), W)
bm = uniform_filter(b.astype(float), W)

amc = am[W // 2 : -W // 2 + 1]
bmc = bm[W // 2 : -W // 2 + 1]

da = a[:, None] - amc
db = b[:, None] - bmc

# Get sliding mask of valid windows
m, n = da.shape
mask1 = np.arange(m)[:, None] >= np.arange(n)
mask2 = np.arange(m)[:, None] < np.arange(n) + W
mask = mask1 & mask2
dam = da * mask
dbm = db * mask

ssAs = np.einsum("ij,ij->j", dam, dam)
ssBs = np.einsum("ij,ij->j", dbm, dbm)
D = np.einsum("ij,ij->j", dam, dbm)
# add np.nan to reach 72
self.cor.append(np.hstack((D / np.sqrt(ssAs * ssBs), np.zeros(9) + np.nan)))