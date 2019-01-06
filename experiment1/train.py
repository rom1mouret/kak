#!/usr/bin/env python3

import sys
import os
import numpy as np

from hyperopt import hp

from shoputils.trainer import run_training
from shoputils.general_net import GeneralNet

cols = ["product", "maker", "brand", "model", "price", "updttm", "img_feat",
        "bcateid", "mcateid", "dcateid", "scateid"]
preproc_cols = ["brand", "maker", "product", "model", "price", "img_feat"]


preproc_files = [sys.argv[1]]
holdout_files = [sys.argv[2]]
optim_files = [sys.argv[3]]
train_files = sys.argv[4:]

# instanciate model
static_params = {
    "vocab_dim": 8192,
    "embed_dim": 1024,
    "latent_dim": 600, # unused
}
model = GeneralNet(static_params, gpu=0, gpu_batch_size=16)

space = {
    "batch_size": hp.qloguniform("batch_size", np.log(64), np.log(700), 32),
    "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
    "lr_embedding": hp.loguniform("lr_embedding", np.log(0.0001), np.log(0.1)),
    "clipping": hp.uniform("clipping", 1.5, 10.0),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
}

first_space_vals = {
    "batch_size": 32,
    "lr_embedding": 0.02,
    "lr": 0.015,
    "clipping": 4.0,
    "momentum": 0.9,
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
