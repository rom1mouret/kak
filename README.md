# kak

** This github project is a work in progress. **
** More documentation will be come later. **
** Code will be refactored. **

## Requirements

* Python >= 3.6 (might work with Python 3.5 as well)
* torch 0.4.1
* hyperopt 0.1.1
* konlpy 0.5.1
* pyhashxx 0.1.3
* pandas (any recent version, tested with 0.22.0)
* numpy (any recent version, tested with 1.14.0)
* PyYAML (any recent version, tested with 3.12)
* tqdm (any recent version, tested with 4.28.1)
* A UNIX system. Hasn't been tested on Windows.
* 3GB GPU
* 16GB RAM

It should work fine with no GPU, though it will be significantly slower.

## shoputils library

All the non-executable code resides in *shoputils* library. It helps with organizing the project into multiple experiments.
Before shuffling and training, install *shoputils*:

```sh
(cd shoputils && make install)
```

About *shoputils* content:

* the most important part is the model itself: shoputils/general_net.py
* roughly 30% of the code is unused. This is because I did other experiments and kept the code around.

## Shuffling

The main training script does not shuffle the data.
If train_dir is the directory where the original training data is, this is how you shuffle it:

```sh
./shuffle.py shuff train_dir/train.chunk.*
```

'shuff' is the prefix of the generated hdf5 files.

time:

* about 2 minutes with no img_feat
* about 5 hours with img_feat included

Current model requires img_feat.

## Training

Start the training with:
```sh
cd experiment1 && ./train.py ../shuff_preproc.h5 ../shuff_holdout.h5 ../shuff_optim.h5 ../shuff_train*
```

You can gracefully terminate the training at any moment by creating an empty file named "stop" in the working directory.
```sh
touch stop
```

time: between 3 hours and 8 hours. Monitor stdout to check the progress.

SGD often gets stuck. When this happens, the script will look for better hyperparameters using Bayesian optimization. This is what really takes time.

When the Bayesian optimization repeatedly fails to improve the evaluation score, it is time to touch *stop*.

## Inference

```sh
./inference.py ../experiment1/models/<last_model>.yml ../dev.chunk.01
```

The TSV results will be created in the same directory.

time: about 20 minutes

## Other scripts

* sample.py
* evaluate.py
