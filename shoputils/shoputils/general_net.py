import yaml
import numpy as np
import re
import os
import time
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bag_of_words import BagOfWords

from .util import (
        to_gpu, split_chunk, slice_tensors,
        accu_loss, consolidate_loss,
        build_vocab, clean_strings, generate_uid,
        yaml_serializable, hash_strings
)


class Net(nn.Module):
    def __init__(self, vocab_dim, embed_dim, latent_dim,
                 b_dim, m_dim, s_dim, d_dim):
        super(Net, self).__init__()

        price_dim = 3
        price_embed_dim = 32
        other_dim = 5
        slope = 0.2
        small_slope = 0.02
        img_dim = 2048
        reduced_img_dim = 128
        latent_dim = embed_dim + price_embed_dim + other_dim + reduced_img_dim

        # text embeddings
        self._embedding = nn.EmbeddingBag(vocab_dim, embed_dim, mode='sum')

        # price embedding (sorta)
        self._price_proc = nn.Sequential(
            nn.Linear(price_dim, 128),
            nn.LeakyReLU(slope),
            nn.Linear(128, price_embed_dim),
            nn.LeakyReLU(slope),
            nn.BatchNorm1d(price_embed_dim)
        )

        # image processing
        self._img_proc = nn.Sequential(
            nn.Linear(img_dim, reduced_img_dim),
            nn.LeakyReLU(slope)
        )

        # residual unit
        self._res = nn.Sequential(
            nn.Linear(latent_dim, 3200),
            nn.LeakyReLU(slope),
            #nn.Dropout(0.2),
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

        leakyrelu_gain = nn.init.calculate_gain("leaky_relu", param=slope)
        for layer in (self._price_proc[0], self._price_proc[2],
                      self._img_proc[0],
                      self._res[0], self._res[2]):
            nn.init.xavier_normal_(layer.weight, gain=leakyrelu_gain)

        leakyrelu_gain2 = nn.init.calculate_gain("leaky_relu", param=small_slope)
        for layer in (self._res[4], ):
            nn.init.xavier_normal_(layer.weight, gain=small_slope)

    def embedding_params(self):
        return self._embedding.parameters()

    def all_params_but_embedding(self):
        layers = (self._price_proc, self._res, self._b, self._m, self._s, self._d)
        params = map(lambda layer: list(layer.parameters()), layers)
        return sum(params, [])

    def forward(self, batch):
        embedded = self._embedding(batch["text"])
        price = self._price_proc(batch["price"])
        img = self._img_proc(batch["img"])
        latent = torch.cat([price, embedded, img, batch["missing"], batch["season"]], dim=1)
        latent = latent + self._res(latent)

        # decision time!
        b = self._b(latent).squeeze(1)
        m = self._m(latent).squeeze(1)
        s = self._s(latent).squeeze(1)
        d = self._d(latent).squeeze(1)

        return b, m, s, d


class GeneralNet:
    def __init__(self, static_params, gpu=-1, gpu_batch_size=16, out_dir="models/"):
        self._gpu = gpu
        self._gpu_batch_size = gpu_batch_size
        trained = type(static_params) is not dict

        if trained:
            self._yaml_path = static_params
            with open(self._yaml_path, "r", encoding="utf-8") as f:
                static_params = yaml.load(f)

            dirloc = os.path.dirname(self._yaml_path)
            self._torch_mdl_path = os.path.join(dirloc, static_params["torch_file"])
            self._vocab_path = os.path.join(dirloc, static_params["vocab_file"])
        else:
            prefix = "general_"
            uid = generate_uid()
            print("creating new model", uid)
            self._torch_mdl_path = os.path.join(out_dir, "%s%s.torch" % (prefix, uid))
            self._yaml_path = os.path.join(out_dir, "%s%s.yml" % (prefix, uid))
            self._vocab_path = os.path.join(out_dir, "vocab_%s.yml" % (uid, ))

        self._static_params = static_params

        # dimensions
        with open("cate1.json", "r") as f:
            content = json.load(f)

        b_dim = max(content["b"].values()) + 1
        m_dim = max(content["m"].values()) + 1
        s_dim = max(content["s"].values()) + 1
        d_dim = max(content["d"].values()) + 1

        print("s_dim", s_dim)
        print("d_dim", d_dim)

        def net_factory():
            net = Net(
                vocab_dim=static_params["vocab_dim"],
                embed_dim=static_params["embed_dim"],
                latent_dim=static_params["latent_dim"],
                b_dim=b_dim,
                m_dim=m_dim,
                s_dim=s_dim,
                d_dim=d_dim
            )

            if self._gpu >= 0:
                net = net.cuda(self._gpu)

            return net

        self._net_factory = net_factory
        self._net = net_factory()

        self._bow = BagOfWords(coarse_cols=("brand", "maker"), other_cols=("product", "model"))

        if trained:
            self.load(self._yaml_path)

    def train_preprocessing(self, chunk):
        # normalization of image features
        print("img feat normalization")
        self._img_mean = chunk["img_feat"].mean(axis=0).astype(np.float32)
        self._img_std = chunk["img_feat"].std(axis=0).astype(np.float32) + 0.000001

        # build vocabulary
        print("building vocabulary")
        bags = self._bow.break_down(chunk)
        counter = Counter()
        for bag in bags:
            counter.update(bag)

        freq = counter.most_common(self._static_params["vocab_dim"]-1)
        self._vocab = {tok: i+1 for i, (tok, c) in enumerate(freq)}

        with open(self._vocab_path, "w", encoding="utf-8") as f:
            yaml.dump(self._vocab, f)

        # prices
        prices = chunk["price"].copy().astype(np.float32)
        missing = prices == -1

        log_prices = np.log(1.01 + prices)
        log_prices[missing] = np.nan
        prices[missing] = np.nan

        self._logprice_median = np.nanmedian(log_prices)
        self._logprice_std = np.nanstd(log_prices)
        self._price_median = np.nanmedian(prices)
        self._price_std = np.nanstd(prices)

        print("price median", self._price_median, "price std", self._price_std)

    def _embedding_input(self, bags):
        # word indices (without the unknown words)
        indices = [
            list(filter(None.__ne__, map(self._vocab.get, bag)))
            for bag in bags
        ]

        # torch tensor
        max_length = max(1, max(map(len, indices)))
        inp = torch.zeros(len(bags), max_length, dtype=torch.int64)
        for i, index_list in enumerate(indices):
            inp[i, :len(index_list)] = torch.LongTensor(index_list)

        return inp

    def _build_batch(self, chunk, with_targets=True, with_idf=True, gpu=-1):
        # targets
        if with_targets:
            b = torch.LongTensor(chunk["bcateid"])
            m = torch.LongTensor(chunk["mcateid"])
            s = torch.LongTensor(chunk["scateid"])
            d = torch.LongTensor(chunk["dcateid"])
            y = to_gpu(gpu, b, m, s, d)
        else:
            y = None

        # embedding feat for text
        bags = self._bow.break_down(chunk)
        text_feats = self._embedding_input(bags)

        # missing columns (with no decoding required)
        missing_product = map(bool, chunk["product"])
        missing_model = map(bool, chunk["product"])
        missing_brand = map(bool, chunk["brand"])
        missing_maker = map(bool, chunk["maker"])
        tranposed = zip(missing_product, missing_model, missing_brand, missing_maker)
        missing = torch.Tensor(list(tranposed))

        # time in the year
        middle = 7 * 30
        days = np.array([int(d[6:8]) for d in chunk["updttm"]], dtype=np.float32)
        months = np.array([int(d[4:6]) for d in chunk["updttm"]], dtype=np.float32)
        season = np.abs(days + 30 * months - middle) /  middle
        season = torch.from_numpy(season).unsqueeze(1)

        # prices
        prices = chunk["price"]
        price_missing = prices == -1
        log_prices = (np.log(1.01 + prices) - self._logprice_median) / self._logprice_std
        log_prices[price_missing] = 0
        reg_prices = (prices - self._price_median) / self._price_std
        reg_prices[price_missing] = 0
        np_price = np.column_stack([price_missing.astype(np.float32), log_prices, reg_prices])
        price_feat = torch.from_numpy(np_price.astype(np.float32))

        # image
        img = torch.as_tensor((chunk["img_feat"] - self._img_mean)/self._img_std, dtype=torch.float32)

        batch = {
            "text": text_feats,
            "price": price_feat,
            "missing": missing,
            "season": season,
            "img": img
        }

        batch = to_gpu(gpu, batch)[0]

        return batch, y

    def prepare_hyperopt_dataset(self, train_set):
        print("building HPO train set")
        self._train_set = self._build_batch(train_set)
        print("HPO train set built")

    def set_holdout_val_set(self, val_set):
        chunks = split_chunk(val_set, batch_size=self._gpu_batch_size,
                             max_rows=5000)
        print("holdout batch size:", len(chunks) * self._gpu_batch_size)
        self._val_batches = list(map(self._build_batch, chunks))

    def train_and_eval(self, space):
        # copy the current net + new optimization hyperparameters
        net = self._net_factory()
        if self._torch_mdl_path is not None:
            net.load_state_dict(torch.load(self._torch_mdl_path))

        optimizers = self._create_optimizer(net, space)

        # training
        train_batch, (b, m, s, d) = self._train_set
        batch_size = int(space["batch_size"])
        L = b.size(0)
        for k in range(0, L, batch_size):
            # forward
            b_loss = [0, 0, 1.0]
            m_loss = [0, 0, 1.2]
            s_loss = [0, 0, 1.3]
            d_loss = [0, 0, 1.4]
            for i in range(k, min(L, k+batch_size), self._gpu_batch_size):
                batch, b_true, m_true, s_true, d_true = \
                    slice_tensors(i, i+self._gpu_batch_size,
                                  train_batch, b, m, s, d)

                if b_true.size(0) == 1:
                    continue # batchnorm wouldn't work

                batch, b_true, m_true, s_true, d_true = \
                    to_gpu(self._gpu, batch, b_true, m_true, s_true, d_true)

                b_pred, m_pred, s_pred, d_pred = net(batch)

                accu_loss(b_pred, b_true, b_loss)
                accu_loss(m_pred, m_true, m_loss)
                accu_loss(s_pred, s_true, s_loss)
                accu_loss(d_pred, d_true, d_loss)

            loss = consolidate_loss(b_loss, m_loss, s_loss, d_loss)

            # gradients
            for optim in optimizers:
                optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self._space["clipping"])

            # weight update
            for optim in optimizers:
                optim.step()

        # evaluation
        return self._validate(net.eval())[0]

    def reset_optimizer(self, space):
        self._optim = self._create_optimizer(self._net, space)
        self._space = space

    def train_on_batch(self, chunk):
        mini_chunks = split_chunk(chunk, batch_size=self._gpu_batch_size)

        # forward
        b_loss = [0, 0, 1.0]
        m_loss = [0, 0, 1.2]
        s_loss = [0, 0, 1.3]
        d_loss = [0, 0, 1.4]
        for mini_chunk in mini_chunks:
            mini_batch, (b, m, s, d) = self._build_batch(mini_chunk, gpu=self._gpu)
            if b.size(0) == 1:
                continue # batchnorm wouldn't work
            b_pred, m_pred, s_pred, d_pred = self._net(mini_batch)
            accu_loss(b_pred, b, b_loss)
            accu_loss(m_pred, m, m_loss)
            accu_loss(s_pred, s, s_loss)
            accu_loss(d_pred, d, d_loss)

        loss = consolidate_loss(b_loss, m_loss, s_loss, d_loss)

        # gradients
        for optim in self._optim:
            optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), self._space["clipping"])

        # weight update
        for optim in self._optim:
            optim.step()

    def predict(self, chunk):
        batch, y = self._build_batch(chunk, gpu=self._gpu, with_targets=False)
        b_pred, m_pred, s_pred, d_pred = self._net(batch)

        pick = lambda vec: vec.argmax(dim=1).data.cpu().numpy()

        return pick(b_pred), pick(m_pred), pick(s_pred), pick(d_pred)

    def save(self, checkpoint, loss, report):
        torch.save(self._net.state_dict(), self._torch_mdl_path)

        metadata = {
            "price_median": self._price_median,
            "price_std": self._price_std,
            "logprice_median": self._logprice_median,
            "logprice_std": self._logprice_std,
            "loss": loss,
            "checkpoint": checkpoint,
            "torch_file": os.path.basename(self._torch_mdl_path),
            "vocab_file": os.path.basename(self._vocab_path),
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "img_mean": self._img_mean.tolist(),
            "img_std": self._img_std.tolist()
        }
        metadata.update(report)
        metadata.update(self._space)
        metadata.update(self._static_params)
        metadata = yaml_serializable(metadata)

        with open(self._yaml_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return self._yaml_path

    def load(self, yaml_file):
        """ build the model structure before calling this method """
        print("loading", yaml_file)
        with open(yaml_file, "r") as f:
            metadata = yaml.load(f)
        self._space = metadata

        # price stuff
        self._price_std = metadata["price_std"]
        self._price_median = metadata["price_median"]
        self._logprice_std = metadata["logprice_std"]
        self._logprice_median = metadata["logprice_median"]

        # image feat normalization
        self._img_mean = np.array(metadata["img_mean"], dtype=np.float32)
        self._img_std = np.array(metadata["img_std"], dtype=np.float32)

        # vocab
        with open(self._vocab_path, "r", encoding="utf-8") as f:
            self._vocab = yaml.load(f)

        # torch network
        self._net.load_state_dict(torch.load(self._torch_mdl_path))

        if self._gpu >= 0:
            self._net = self._net.cuda(self._gpu)

    def eval(self):
        self._net = self._net.eval()
        return self

    def validate(self):
        net = self._net.eval()
        val = self._validate(net)
        self._net.train()

        return val

    def _create_optimizer(self, net, space):
        #optim = torch.optim.SGD(
        #     net.parameters(),
        #     lr=space["lr"],
        #     momentum=space["momentum"],
        # )
        # return (optim, )

        optim1 = torch.optim.SGD(
            net.embedding_params(),
            lr=space["lr_embedding"],
            momentum=space["momentum"],
            weight_decay=0)

        optim2 = torch.optim.SGD(
            net.all_params_but_embedding(),
            lr=space["lr"],
            momentum=space["momentum"],
            weight_decay=0.000001)

        # optim2 = torch.optim.Adam(
        #     net.all_params_but_embedding(),
        #     lr=space["lr"],
        #     weight_decay=0.000001)

        return optim1, optim2

    def _validate(self, net):
        b_loss = [0, 0, 1.0]
        m_loss = [0, 0, 1.2]
        s_loss = [0, 0, 1.3]
        d_loss = [0, 0, 1.4]
        m_acc = 0
        b_acc = 0
        s_acc = 0
        d_acc = 0
        for batch, (b, m, s, d) in self._val_batches:
            batch, b, m, s, d = to_gpu(self._gpu, batch, b, m, s, d)
            b_pred, m_pred, s_pred, d_pred = net(batch)
            m_acc += (m_pred.argmax(dim=1) == m).float().sum().item()
            b_acc += (b_pred.argmax(dim=1) == b).float().sum().item()
            s_acc += (s_pred.argmax(dim=1) == s).float().sum().item()
            d_acc += (d_pred.argmax(dim=1) == d).float().sum().item()

            accu_loss(b_pred, b, b_loss)
            accu_loss(m_pred, m, m_loss)
            accu_loss(s_pred, s, s_loss)
            accu_loss(d_pred, d, d_loss)

        loss = consolidate_loss(b_loss, m_loss, s_loss, d_loss).item()

        b_acc /= b_loss[1].item()
        m_acc /= m_loss[1].item()
        s_acc /= s_loss[1].item()
        d_acc /= d_loss[1].item()

        tot_acc = (b_acc + 1.2 * m_acc + 1.3 * s_acc + 1.4 * d_acc) / 4

        report = {
            "loss": loss,
            "b_acc": b_acc,
            "m_acc": m_acc,
            "s_acc": s_acc,
            "d_acc": d_acc,
            "acc": tot_acc
        }

        metric = loss/50 + 1.25 - tot_acc

        return metric, report
