#!/usr/bin/env python3

import h5py
import sys
import numpy as np
import time
from sklearn.utils import shuffle
import pandas as pd
from multiprocessing import Pool


cols = ["bcateid", "mcateid", "scateid", "dcateid", "price", "img_feat"
        "product", "brand", "maker", "model", "updttm"]

prefix = sys.argv[1]
filenames = sys.argv[2:]

def find_group(f):
    if "train" in f:
        return f["train"]
    if "dev" in f:
        return f["dev"]
    return f["test"]

# collect the number of rows
n_rows = []
for filename in filenames:
    with h5py.File(filename, "r") as f:
        grp = find_group(f)
        n_rows.append(len(grp["price"]))

offsets = np.cumsum(n_rows) - np.array(n_rows)
tot_rows = sum(n_rows)
print("total number of rows:", tot_rows)


def generate_set(args):
    output_path, indices = args
    print("generating", output_path)

    indices.sort()
    big_chunk = {col: [] for col in cols}
    for j, (offset, size) in enumerate(zip(offsets, n_rows)):
        in_file = indices[(indices >= offset) & (indices < (offset+size))] - offset

        # transform to mask because it is faster
        mask = np.zeros(size, dtype=bool)
        mask[in_file] = True

        # retrieve the data
        with h5py.File(filenames[j], "r") as f:
            grp = find_group(f)
            for col in cols:
                if col == "img_feat":
                    big_chunk[col].append(grp[col][mask, :])
                else:
                    big_chunk[col].append(grp[col][mask])

    # concatenate in a new hdf5 file
    print("writing in", output_path)
    random_order = np.random.permutation(len(indices))
    with h5py.File(output_path, "w") as f:
        grp = f.create_group("train")
        for col in cols:
            arr = np.concatenate(big_chunk[col], axis=0)
            grp[col] = arr.take(indices=random_order, axis=0)


all_indices = np.random.permutation(tot_rows)

# holdout validation set
holdout_cut = 5000
holdout_indices = all_indices[:holdout_cut]
inp = [("%s_holdout.h5" % prefix, holdout_indices)]

# Bayesian optim set
optim_cut = holdout_cut + 60000
optim_indices = all_indices[holdout_cut:optim_cut]
inp += [("%s_optim.h5" % prefix, optim_indices)]

# other sets
n = (tot_rows - optim_cut) // 1000000
arrs = np.array_split(all_indices[optim_cut:], n)
names = ("%s_train%i.h5" % (prefix, i+1) for i in range(n))
inp += list(zip(names, arrs))

# preproc sets with overlapping indices
preproc_indices = np.random.permutation(tot_rows)[:100000]
inp += [("%s_preproc.h5" % prefix, preproc_indices)]

# train sets
Pool(8).map(generate_set, inp)
