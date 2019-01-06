#!/usr/bin/env python3

import logging
import sys
import os
import numpy as np

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_FAIL, STATUS_OK, Trials
from shoputils.hdf5_io import read_hdf5_sets


def init_model(
    model,
    holdout_files,
    optim_files,
    preproc_files,
    train_files,
    cols,
    preproc_cols):

    assert len(train_files) >= 1, "no training files found"
    assert len(optim_files) >= 1, "no Bayesian optimization files found"
    assert len(holdout_files) >= 1, "no holdout validation files found"
    assert len(preproc_files) >= 1, "no preprocessing files found"

    print("training files", train_files)
    print("holdout files", holdout_files)
    print("preproc files", preproc_files)
    print("Bayesian optim files", optim_files)

    # read holdout & Bayesian optim & preproc data
    for holdout_chunk in read_hdf5_sets(holdout_files, chunk_size=5000, cols=cols):
        break

    for optim_chunk in read_hdf5_sets(optim_files, chunk_size=50000, cols=cols):
        break

    for preproc_chunk in read_hdf5_sets(preproc_files, chunk_size=100000, cols=preproc_cols):
        break

    model.train_preprocessing(preproc_chunk)
    model.prepare_hyperopt_dataset(optim_chunk)
    model.set_holdout_val_set(holdout_chunk)

def run_training(
    model,
    space,
    space_values,
    holdout_files,
    optim_files,
    preproc_files,
    train_files,
    cols,
    preproc_cols):

    assert "batch_size" in space_values

    try:
        os.remove("stop")
    except:
        pass
    else:
        print("removed 'stop' file")

    init_model(model, holdout_files, optim_files, preproc_files,
               train_files, cols, preproc_cols)

    best_loss = np.inf

    # Bayesian optimization stuff
    def objective(space_values):
        print("values", space_values)
        if os.path.exists("stop"):
            return {"status": STATUS_FAIL}

        loss = []
        try:
            for seed in range(3):
                np.random.seed(seed) # it is better to compare the algorithms with the same data
                l = model.train_and_eval(space_values)
                loss.append(l)
                print("hyperopt loss", l, "VS best_loss", best_loss)
                if (np.mean(loss) >= 1.1 * best_loss and len(loss) >= 2) or \
                    l >= 2 * best_loss:
                    break  # no need to search further
        except:
            logging.exception("cannot call eval") # most likely a CUDA OOM
            return {"status": STATUS_FAIL}

        var = np.var(loss)
        std = np.sqrt(var + 0.0000001)
        metric = np.mean(loss) + 0.5 * std  # pessimistic

        return {"status": STATUS_OK,
                "loss": float(metric),
                "loss_variance": float(var)}

    # main training
    checkpoint = (train_files[0], 0)
    patience_init = 14
    patience = patience_init  # as per early-stopping terminology
    model.reset_optimizer(space_values)

    while not os.path.exists("stop"):
        for iteration, (chunk, current_loc) in enumerate(read_hdf5_sets(
                                                        train_files,
                                                        chunk_size=space_values["batch_size"],
                                                        cols=cols,
                                                        loc=checkpoint)):
            if os.path.exists("stop"):
                break

            # training
            model.train_on_batch(chunk)

            # validation
            if iteration % 30 == 0:
                loss, report = model.validate()
                better = loss < best_loss
                report_str = "; ".join(("%s: %.3f" % k_v for k_v in report.items()))
                print("metric: %.3f; %s; improvement: %s" % (loss, report_str, better))

                if better:
                    best_loss = loss
                    yaml_file = model.save(current_loc, loss, report)
                    patience = patience_init
                    trials = Trials()
                    checkpoint = current_loc
                elif patience == 0:
                    # optimize the hyperparameters using best model so far
                    model.load(yaml_file)
                    for repeat in range(6):
                        space_values = fmin(objective, space,
                                            algo=tpe.suggest,
                                            max_evals=len(trials) + 30 - 4 * repeat,
                                            trials=trials)
                        min_loss = min(filter(None, trials.losses()))
                        threshold = 1.0 * best_loss
                        print("min loss:", min_loss, "vs", threshold)
                        if min_loss < threshold or os.path.exists("stop"):
                            break

                    # reset the last best bodel with new hyperparameters
                    model.reset_optimizer(space_values)
                    patience_init += 2
                    patience = patience_init
                    print("best", space_values)
                    break
                else:
                    patience -= 1

    return yaml_file
