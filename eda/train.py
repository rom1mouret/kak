#!/usr/bin/env python3

import sys
import os
import numpy as np

from hyperopt import hp

from shoputils.hdf5_io import read_hdf5_sets
from shoputils.trainer import run_training
from shoputils.data_fitter import DataFitter

cols = ["maker", "brand", "price", "product", "model"]
preproc_cols = ["price", "brand", "maker"]

# locate the training/testing data
data_dir = sys.argv[1]
holdout_files = []
preproc_files = []
optim_files = []
train_files = []
for fname in os.listdir(data_dir):
    path = os.path.join(data_dir, fname)
    if "train" in fname:
        train_files.append(path)
    elif "holdout" in fname:
        holdout_files.append(path)
    elif "preproc" in fname:
        preproc_files.append(path)
    elif "optim" in fname:
        optim_files.append(path)

threshold_files = [train_files[-1]]
train_files = train_files[:-1]

# instanciate model
static_params = {
    "text_input_dim": 16384,
    "text_embed_dim": 1024,
    "text_output_dim": 4096,
    "brand_dim": 256,
    "latent_dim": 320
}
model = DataFitter(static_params, gpu=0)

space = {
    "batch_size": hp.qloguniform("batch_size", np.log(32), np.log(1024), 32),
    "lr": hp.loguniform("lr", np.log(0.00001), np.log(0.05)),
    "clipping": hp.uniform("clipping", 1.5, 10.0),
    "momentum": hp.uniform("momentum", 0.3, 0.95)
}

first_space_vals = {
    "batch_size": 64,
    "lr": 0.015,
    "clipping": 4.0,
    "momentum": 0.9
}

run_training(
    model=model,
    space=space,
    space_values=first_space_vals,
    holdout_files=holdout_files,
    optim_files=optim_files,
    preproc_files=preproc_files,
    train_files=train_files,
    cols=cols,
    preproc_cols=preproc_cols
)

for chunk in read_hdf5_sets(threshold_files, chunk_size=30000, cols=cols):
    break

model.find_err_thresholds(chunk)
