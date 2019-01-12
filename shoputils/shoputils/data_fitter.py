import yaml
import numpy as np
import re
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import (
        to_gpu, split_chunk, slice_tensors,
        build_vocab, clean_strings, generate_uid,
        yaml_serializable, hash_strings
)


class Net(nn.Module):
    def __init__(self, text_input_dim, text_embed_dim, text_output_dim,
                 brand_dim, latent_dim):
        super(Net, self).__init__()

        self._latent_dim = latent_dim
        price_dim = 3
        price_embed_dim = 8
        other_dim = 4
        slope = 0.2

        hidden_dim = 2 * self._latent_dim
        self._dim = self._latent_dim + 2 * hidden_dim + price_embed_dim

        # product + model embeddings
        self._embedding = nn.EmbeddingBag(text_input_dim, text_embed_dim, mode='sum')

        # price embedding (sorta)
        self._price_proc = nn.Sequential(
            nn.Linear(price_dim, 128),
            nn.LeakyReLU(slope),
            nn.Linear(128, price_embed_dim)
        )

        # encoder
        self._encoder = nn.Sequential(
            nn.Linear(text_embed_dim + price_embed_dim + other_dim, 2048),
            nn.LeakyReLU(slope),
            nn.Linear(2048, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, self._latent_dim),
        )

        # decoders
        self._txt_decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(slope),
            nn.Dropout(0.25),
            nn.Linear(2048, text_output_dim)
        )

        self._brand_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, brand_dim),
            nn.LogSoftmax()
        )

        self._maker_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, brand_dim),
            nn.LogSoftmax()
        )

        for layer in (self._price_proc[0],
                      self._encoder[0], self._encoder[2],
                      self._txt_decoder[0],
                      self._brand_decoder[0],  self._maker_decoder[0]):
            nn.init.xavier_normal_(
                layer.weight,
                gain=nn.init.calculate_gain("leaky_relu", param=slope))

    def dim(self):
        return self._dim

    def extract_features(self, batch):
        latent, price = self.latent_features(batch)
        b = F.linear(latent, self._brand_decoder[0].weight)
        m = F.linear(latent, self._maker_decoder[0].weight)

        return torch.cat([latent, price, b, m], dim=1)

    def latent_features(self, batch):
        # embeddings
        embedded = self._embedding(batch["text"])

        # price pre-processing
        price = self._price_proc(batch["price"])

        # encoder input
        inp = torch.cat([embedded, price, batch["missing"]], dim=1)

        # encoder
        encoded = self._encoder(inp)

        return encoded, price

    def forward(self, batch):
        latent = self.latent_features(batch)[0]
        txt = self._txt_decoder(latent)
        brand = self._brand_decoder(latent)
        maker = self._maker_decoder(latent)

        return txt, brand, maker


class DataFitter:
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
            prefix = "fitter_"
            uid = generate_uid()
            print("creating new model", uid)
            self._torch_mdl_path = os.path.join(out_dir, "%s%s.torch" % (prefix, uid))
            self._yaml_path = os.path.join(out_dir, "%s%s.yml" % (prefix, uid))
            self._vocab_path = os.path.join(out_dir, "vocab_%s.yml" % (uid, ))

        self._static_params = static_params

        def net_factory():
            net = Net(
                text_input_dim=static_params["text_input_dim"],
                text_embed_dim=static_params["text_embed_dim"],
                text_output_dim=static_params["text_output_dim"],
                brand_dim=static_params["brand_dim"],
                latent_dim=static_params["latent_dim"]
            )

            if self._gpu >= 0:
                net = net.cuda(self._gpu)

            return net

        self._net_factory = net_factory
        self._net = net_factory()
        self._bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self._nll_loss = nn.NLLLoss(reduction='sum')

        self._foreign = re.compile("[a-z]+")
        self._hangul = re.compile("[가-힣]")
        self._number = re.compile("[0-9]+")

        if trained:
            self.load(self._yaml_path)

    def __getitem__(self, key):
        return self._static_params[key]

    def train_preprocessing(self, chunk):
        # brand & maker
        vocab_dim = self._static_params["brand_dim"]
        self._brand_vocab = build_vocab(chunk["brand"], vocab_dim, shift=1)
        self._maker_vocab = build_vocab(chunk["maker"], vocab_dim, shift=1)

        vocab = {
            "brand": self._brand_vocab,
            "maker": self._maker_vocab
        }

        with open(self._vocab_path, "w", encoding="utf-8") as f:
            yaml.dump(vocab, f)

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

    def _tokenize(self, text):
        """ gets the text input ready for embedding """
        text = text.lower()
        arr = list(self._hangul.findall(text))
        arr += list(self._foreign.findall(text))
        if self._number.search(text):
            arr.append("9")

        return set(arr)

    def _text_output(self, sequences):
        vocab_dim = self._static_params["text_output_dim"]
        X = torch.zeros(len(sequences), vocab_dim)
        for i, seq in enumerate(sequences):
            indices = hash_strings(seq, mod=vocab_dim, seed=1)
            if len(indices) > 0:
                X[[i] * len(indices), indices] = 1

        return X

    def _embedding_input(self, sequences):
        vocab_dim = self._static_params["text_input_dim"]
        max_length = max(1, max(map(len, sequences)))
        inp = torch.zeros(len(sequences), max_length, dtype=torch.int64)
        for i, seq in enumerate(sequences):
            indices = hash_strings(seq, mod=vocab_dim-1, seed=0) + 1
            inp[i, :len(indices)] = torch.LongTensor(indices)

        return inp

    def _build_batch(self, chunk, with_targets=True, with_idf=True, gpu=-1):
        cols = {}  # to store a copy of the chunk

        # decoding
        for col_name in ("product", "model"):
            cols[col_name] = (s.decode("utf-8") for s in chunk[col_name])

        # tokenization
        sequences = []
        for product, model in zip(cols["product"], cols["model"]):
            sequences.append(self._tokenize(product) | self._tokenize(model))

        # text input
        text = self._embedding_input(sequences)

        # missing columns (with no decoding required)
        missing_product = map(bool, chunk["product"])
        missing_model = map(bool, chunk["product"])
        missing_brand = map(bool, chunk["brand"])
        missing_maker = map(bool, chunk["maker"])
        tranposed = zip(missing_product, missing_model, missing_brand, missing_maker)
        missing = torch.Tensor(list(tranposed))

        # prices
        prices = chunk["price"]
        price_missing = prices == -1
        log_prices = (np.log(1.01 + prices) - self._logprice_median) / self._logprice_std
        log_prices[price_missing] = 0
        reg_prices = (prices - self._price_median) / self._price_std
        reg_prices[price_missing] = 0
        np_price = np.column_stack([price_missing.astype(np.float32), log_prices, reg_prices])
        price_feat = torch.from_numpy(np_price.astype(np.float32))

        if with_targets:
            # brand & makers
            for col_name in ("brand", "maker"):
                cols[col_name] = clean_strings((s.decode("utf-8") for s in chunk[col_name]))

            # maker & brand
            makers = torch.LongTensor([self._maker_vocab.get(v, 0) for v in cols["maker"]])
            brands = torch.LongTensor([self._brand_vocab.get(v, 0) for v in cols["brand"]])

            # text
            if with_idf:
                y_text = self._text_output(sequences)
            else:
                y_text = None

            # move everything to GPU
            y = to_gpu(gpu, y_text, makers, brands)
        else:
            y = None

        batch = {
            "text": text,
            "price": price_feat,
            "missing": missing
        }

        batch = to_gpu(gpu, batch)[0]

        return batch, y

    def extract_features(self, chunk):
        batch, y = self._build_batch(chunk, with_targets=False, gpu=self._gpu)
        return self._net.extract_features(batch).data

    def prediction_err(self, chunk):
        batch, (y, brand, maker) = self._build_batch(chunk, with_targets=True, gpu=self._gpu)
        y_pred, brand_pred, maker_pred = self._net(batch)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        nll = nn.NLLLoss(reduction='none')
        err1 = bce(y_pred, y).sum(dim=1).data
        err2 = nll(brand_pred, brand).data
        err3 = nll(maker_pred, maker).data

        return err1, err2, err3

    def find_err_thresholds(self, chunk):
        self._net.eval()
        minichunks = split_chunk(chunk, batch_size=self._gpu_batch_size)
        errs1, errs2, errs3 = [], [], []
        for minichunk in minichunks:
            err1, err2, err3 = self.prediction_err(minichunk)
            errs1 += err1.tolist()
            errs2 += err2.tolist()
            errs3 += err3.tolist()

        errs = np.array([errs1, errs2, errs3])

        thresholds = {}
        for p in (50, 75, 80, 85, 90, 92, 95, 96, 97, 98, 99):
            thresholds[p] = np.percentile(errs, p, axis=1).tolist()

        # save in yaml file
        with open(self._yaml_path, "r", encoding="utf-8") as f:
            content = yaml.load(f)
        content["thresholds"] = thresholds
        with open(self._yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False)

        self._net.train()

        return thresholds

    def dim(self):
        return self._net.dim()

    def prepare_hyperopt_dataset(self, train_set):
        self._train_set = self._build_batch(train_set)

    def set_holdout_val_set(self, val_set):
        chunks = split_chunk(val_set, batch_size=self._gpu_batch_size,
                             max_rows=5000)
        print("holdout batch size:", len(chunks) * self._gpu_batch_size)
        self._val_batches = list(map(self._build_batch, chunks))

    def train_and_eval(self, space):
        # copy the current net + new optimization hyperparameters
        net = self._net_factory()
        if os.path.exists(self._torch_mdl_path):
            net.load_state_dict(torch.load(self._torch_mdl_path))

        optim = self._create_optimizer(net, space)

        # training
        train_batch, (y, brand, maker) = self._train_set
        batch_size = int(space["batch_size"])
        L = y.size(0)
        ordering = np.random.permutation(L)  # gets us a meaningful variance to provide to Bayesian optim
        for k in range(0, L, batch_size):
            # forward
            loss = 0
            for i in range(k, min(L, k+batch_size), self._gpu_batch_size):
                indices = ordering[i:i+self._gpu_batch_size]
                batch, y_true, brand_true, maker_true = \
                    slice_tensors(indices, train_batch, y, brand, maker)
                batch, y_true, brand_true, maker_true = \
                    to_gpu(self._gpu, batch, y_true, brand_true, maker_true)

                # prediction
                y_pred, brand_pred, maker_pred = net(batch)

                # loss
                loss = loss + self._bce_loss(y_pred, y_true) + \
                              self._nll_loss(brand_pred, brand_true) + \
                              self._nll_loss(maker_pred, maker_true)

            # backward
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), space["clipping"])
            optim.step()

        # evaluation
        return self._validate(net.eval())[0]

    def reset_optimizer(self, space):
        self._optim = self._create_optimizer(self._net, space)
        self._space = space

    def train_on_batch(self, chunk):
        mini_chunks = split_chunk(chunk, batch_size=self._gpu_batch_size)

        # forward
        loss = 0
        for mini_chunk in mini_chunks:
            mini_batch, (y_true, maker_true, brand_true) = \
                self._build_batch(mini_chunk, gpu=self._gpu)
            y_pred, maker_pred, brand_pred = self._net(mini_batch)
            loss = loss + self._bce_loss(y_pred, y_true) + \
                          self._nll_loss(maker_pred, maker_true) + \
                          self._nll_loss(brand_pred, brand_true)

        # backward
        self._optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), self._space["clipping"])
        self._optim.step()

    def save(self, checkpoint, loss, report):
        torch.save(self._net.state_dict(), self._torch_mdl_path)

        metadata = {
            "price_median": self._price_median,
            "price_std": self._price_std,
            "logprice_median": self._logprice_median,
            "logprice_std": self._logprice_std,
            "feat_extract_dim": self._net.dim(),
            "loss": loss,
            "checkpoint": checkpoint,
            "torch_file": os.path.basename(self._torch_mdl_path),
            "vocab_file": os.path.basename(self._vocab_path),
            "time": time.strftime("%Y-%m-%d %H:%M")
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

        if "thresholds" in metadata:
            self._thresholds = metadata["thresholds"]

        # price stuff
        self._price_std = metadata["price_std"]
        self._price_median = metadata["price_median"]
        self._logprice_std = metadata["logprice_std"]
        self._logprice_median = metadata["logprice_median"]

        # vocab
        with open(self._vocab_path, "r", encoding="utf-8") as f:
            vocab = yaml.load(f)
        self._brand_vocab = vocab["brand"]
        self._maker_vocab = vocab["maker"]

        # torch network
        self._net.load_state_dict(torch.load(self._torch_mdl_path))

        if self._gpu >= 0:
            self._net = self._net.cuda(self._gpu)

    def thresholds(self):
        return self._thresholds

    def eval(self):
        self._net = self._net.eval()
        return self

    def validate(self):
        net = self._net.eval()
        val = self._validate(net)
        self._net.train()

        return val

    def _create_optimizer(self, net, space):
        optim = torch.optim.SGD(
            net.parameters(),
            lr=space["lr"],
            momentum=space["momentum"],
            weight_decay=0)

        return optim

    def _validate(self, net):
        loss = 0
        n = 0
        for batch, (y_true, maker_true, brand_true) in self._val_batches:
            batch, y_true, maker_true, brand_true = \
                to_gpu(self._gpu, batch, y_true, maker_true, brand_true)
            y_pred, maker_pred, brand_pred = net(batch)
            loss += self._bce_loss(y_pred, y_true) + \
                    self._nll_loss(maker_pred, maker_true) + \
                    self._nll_loss(brand_pred, brand_true)
            n += y_pred.size(0)

        loss /= 3 * n

        return loss.item(), {}
