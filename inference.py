#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import yaml
import os

from shoputils.hdf5_io import read_hdf5_sets
from shoputils.general_net import GeneralNet
from shoputils.util import split_chunk


model_fname = sys.argv[1]
data_files = sys.argv[2:]

classes = ("bcateid", "mcateid", "scateid", "dcateid")
cols = ["maker", "brand", "price", "product", "model", "pid", "updttm", "img_feat"]

gpu = 0
batch_size = 64
model = GeneralNet(model_fname, gpu=gpu, gpu_batch_size=batch_size)

pred_list = []
for i, chunk in enumerate(read_hdf5_sets(
                            set_files=data_files,
                            chunk_size=batch_size,
                            key="dev",  # replace with 'test' for final test set
                            cols=cols)):

    pids = (x.decode() for x in chunk["pid"])
    b_pred, m_pred, s_pred, d_pred = model.predict(chunk)
    pred_list += list(zip(b_pred, m_pred, s_pred, d_pred, pids))

    if i % 100 == 0:
        print("chunk", i)

# write in a CSV
pred_filename = os.path.basename(model_fname).replace(".yml", "") + "_pred.tsv"
df = pd.DataFrame(pred_list, columns=["b", "m", "s", "d", "pid"])
df.to_csv(pred_filename, header=False, index=False, sep="\t", columns=["pid", "b", "m", "s", "d"])
