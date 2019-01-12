#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import os
from collections import defaultdict

from shoputils.data_fitter import DataFitter
from shoputils.hdf5_io import read_hdf5_sets

key = sys.argv[1]
dataset = sys.argv[2:]
cols = ["product", "model", "brand", "maker", "price", "updttm"]

# instanciate sampler
batch_size = 64
fitter = DataFitter("models/fitter_0102_0140_39.yml", gpu=0, gpu_batch_size=batch_size)
fitter.eval()

thresholds = fitter.thresholds()[99]

total_err = defaultdict(float)
total_row = defaultdict(int)

for chunk in read_hdf5_sets(dataset, chunk_size=batch_size, cols=cols, key=key):
    # extract year & month
    decoded = (d.decode() for d in chunk["updttm"])
    times = [13 * int(d[0:4]) + int(d[4:6]) for d in decoded]

    # difference of distribution
    err1, err2, err3 = fitter.prediction_err(chunk)
    err1 = np.clip(err1, 0, thresholds[0])
    err2 = np.clip(err2, 0, thresholds[1])
    err3 = np.clip(err1, 0, thresholds[2])
    err = (np.column_stack([err1, err2, err3])/thresholds).mean(axis=1)

    # statistics
    for year_month, err_value in zip(times, err):
        total_err[year_month] += err_value
        total_row[year_month] += 1

# reporting
vals = []
for key, v in total_err.items():
    month = key % 13
    year = key / 13
    sim = 1 - v/total_row[key]
    vals.append("%i %2i: %.3f" % (year, month, sim))


vals.sort()
for s in vals:
    print(s)
