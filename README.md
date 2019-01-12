# kak

This project is a solution to Kakao's shopping mall ML competition.
https://arena.kakao.com/c/1/

## Shameless plug

You want to play around with CNNs, LSTMs, GANs in a Seoul-based company? hit me up at ai@igloosec.com (CC: mouret.romain) with a resume ;)

For any question about the project, submit a github issue.

## Requirements

* Python >= 3.6 (might work with Python 3.5 as well)
* torch 0.4.1
* hyperopt 0.1.1
* konlpy 0.5.1
* pyhashxx 0.1.3
* h5py 2.8.0
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

## Algorithm

Parameters are optimized with two SGDs: one for the embeddings, one for the rest of the network.
Learning rate, momentum and gradient clipping are tuned using Bayesian optimization in shoputils/trainer.py.
As for the the architecture of the neural network, it is pretty basic:

```python

self._embedding = nn.EmbeddingBag(vocab_dim, embed_dim, mode='sum')

# image processing
self._img_proc = nn.Sequential(
    nn.Linear(img_dim, reduced_img_dim),
    nn.LeakyReLU(slope)
)

# residual unit
self._res = nn.Sequential(
    nn.Linear(latent_dim, 3200),
    nn.LeakyReLU(slope),
    nn.Linear(3200, 2 * latent_dim),
    nn.LeakyReLU(slope),
    nn.Linear(2 * latent_dim, latent_dim),
    nn.LeakyReLU(small_slope),
    nn.BatchNorm1d(latent_dim)
)

# decision layers
self._b = nn.Sequential(
    nn.Linear(latent_dim, b_dim),
    nn.LogSoftmax(dim=1)
)

self._m = nn.Sequential(
    nn.Linear(latent_dim, m_dim),
    nn.LogSoftmax(dim=1)
)

self._s = nn.Sequential(
    nn.Linear(latent_dim, s_dim),
    nn.LogSoftmax(dim=1)
)

self._d = nn.Sequential(
    nn.Linear(latent_dim, d_dim),
    nn.LogSoftmax(dim=1)
)

```
(a couple of layers are not shown here for the sake of brevity)


## Shuffling

The main training script does not shuffle the data.
If train_dir is the directory where the original training data is, this is how you shuffle it:

```sh
./shuffle.py shuff train_dir/train.chunk.*
```

'shuff' is the prefix of the generated hdf5 files.

time:

* about 2 minutes with no img_feat
* about 5 hours with img_feat included (sorry about that)

Current model requires img_feat.

## Exploratory Data Analysis (EDA)

Some competitors have noticed a gap between the leaderboard score and their holdout score.
This is explained by the difference of distribution between dev sets and train sets.

First off, most of the data in the dev set is late 2018, whereas the bulk of the train set is early 2018. So obviously there will be a difference.
But how different?

To answer this question, I ran a mixture of autoencoding and self-supervised learning (shoputils/data_fitter.py) on the dev set. Then I applied the model to the train set (eda/similarity.py).
The higher the more similar:

date | dev.chunk.01 | train.chunk.01 | train.chunk.03 | train.chunk.05
------------ | ------------- | ------------- | ------------- | -------------
2017 10 | 0.762  | 0.723 | 0.756 | 0.757
2017 11 | 0.744 | 0.721 | 0.743 | 0.738
2017 12 | 0.744 | 0.720 | 0.421 | 0.422
2018 01 | 0.724 | 0.711 | 0.556 | 0.557
2018 02 | 0.795 | 0.747 | 0.739 | 0.740
2018 03 | 0.738 | 0.731 | 0.717 | 0.716
2018 04 | 0.738 | 0.731 | 0.724 | 0.725
2018 05 | 0.844 | - | - | -
2018 06 | 0.781 | - | - | -
2018 07 | 0.815 | - | - | -
2018 08 | 0.800 | - | - | -
2018 09 | 0.811 | - | - | -
2018 10 | 0.789 | - | - | -

So what is going on here? The model is marginally better at fitting late 2018 data: 0.8 VS 0.725-0.795. I don't think the difference is large enough to warrant using a sophisticated technique such as transductive transfer learning or meta-learning. It seems reasonable to prevent overfitting the old-fashion way: regularization, dropout and an economical number of parameters.

Also worth mentioning: dev.chunk.01 is unsurprisingly more similar to late 2017 than early 2018. Winter shopping != Spring shopping. Perhaps the model could benefit from seasonal features.

But there is more: while train.chunk.01 is fairly similar to dev.chunk.01 all year long, train.chunk.0[35] are shockingly different in late 2017.
It is safe to assume that the training files do not follow the same, unique distribution. In particular, train.chunk.01 is skewed towards dev.chunk.01, although not quite the same.

This is not really news. The distribution of categories is also different across the training files. For example, these are the most frequent bcateids in train.chunk.01 and their corresponding frequencies in train.chunk.0[35]:

bcateid | train.chunk.01 | train.chunk.03 | train.chunk.05
------------ | ------------- | ------------- | -------------
14  |  100729 | 62164 | 61768
9  |    87815 | 49624 | 49341
20   |  63878 | 28055 | 28081
7   |   61044 | 27422 | 27552
6   |   56880 | 31957 | 31896
17  |   50599 | 48795 | 17158
22   |  45166 | 18825 | 18878

The similarity between train.chunk.03 and train.chunk.05 leaves no doubt that this is not a statistical glitch.

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
