import re
import torch
from datetime import datetime
import numpy as np
from collections import Counter
from pyhashxx import hashxx

cleaning_re = re.compile("[^a-z가-힣]")

text_cols = set(["product", "model", "maker", "brand"])


def clean_strings(vals):
    return (cleaning_re.sub("", v.lower()) for v in vals)


def _mv_to_gpu(element, gpu):
    if element is None:
        return None
        
    #TODO: do this async using os.environ
    if type(element) is dict:
        return {k: v.cuda(gpu) for k, v in element.items()}
    if type(element) is list:
        return [v.cuda(gpu) for v in element]

    return element.cuda(gpu)


def to_gpu(gpu, *args):
    if gpu >= 0:
        arr = [_mv_to_gpu(item, gpu) for item in args]
    else:
        arr = list(args)
    return arr


def _slice(element, i, j):
    if type(element) is dict:
        return {k: t[i:j].clone() for k, t in element.items()}
    if type(element) is list:
        return [t[i:j].clone() for t in element]

    return element[i:j].clone()


def _indexing(element, indices):
    if type(element) is dict:
        return {k: t[indices].clone() for k, t in element.items()}
    if type(element) is list:
        return [t[indices].clone() for t in element]

    return element[indices].clone()


def slice_tensors(i, j, *args):
    if type(i) is int:
        return [_slice(tensor, i, j) for tensor in args]
    else:
        args = [j] + list(args)
        return [_indexing(tensor, i) for tensor in args]

def device(tensor):
    if tensor.device.type == "cpu":
        return -1

    return tensor.device.index


first = True

def split_chunk(chunk, batch_size, max_rows=-1, key="product"):
    for col in chunk.values():
        tot_rows = len(col)
        break

    if max_rows == -1:
        max_rows = tot_rows

    if key not in chunk or key is None:
        global first
        if first:
            print("warning: key cannot be found in chunk. keys:", list(chunk.keys()))
            first = False
        ordering = np.arange(max_rows)
    else:
        lengths = list(map(len, chunk[key]))
        lengths = lengths[:max_rows]
        ordering = np.argsort(lengths)

    chunks = []
    n = len(ordering)
    for i in range(0, n, batch_size):
        indices = ordering[i:i+batch_size]
        mini_chunk = {
            col: [vals[j] for j in indices] if type(col) is list else vals[indices]
            for col, vals in chunk.items()
        }
        chunks.append(mini_chunk)

    return chunks


def shopping_loss(b, m, s, d, b_true, m_true, s_true, d_true):
    """ deprecated """
    cross_entropy = torch.nn.NLLLoss()

    s_cond = (s_true != -1)
    d_cond = (d_true != -1)

    components = 2
    loss = 1.0 * cross_entropy(b, b_true) + 1.2 * cross_entropy(m, m_true)

    if s_cond.sum() > 0:
        loss = loss + 1.3 * cross_entropy(s[s_cond], s_true[s_cond])
        components += 1

    if d_cond.sum() > 0:
        loss = loss + 1.4 * cross_entropy(d[d_cond], d_true[d_cond])
        components += 1

    return loss / components


def accu_loss(y_pred, y_true, loss_result, loss_func=torch.nn.NLLLoss(reduction='sum')):
    cond = (y_true != -1)
    cond_sum = cond.float().sum()
    if cond_sum == 0:
        return

    loss_result[0] += loss_func(y_pred[cond], y_true[cond])
    loss_result[1] += cond_sum


def consolidate_loss(*loss_triplets):
    loss = 0
    for v, count, weight in loss_triplets:
        if count > 0:
            loss = loss + weight * v / count

    return loss


def build_vocab(col_vals, vocab_size, shift=0, decode=True, clean=True):
    if decode:
        col_vals = (v.decode("utf-8") for v in col_vals)
    if clean:
        col_vals = clean_strings(col_vals)

    counter = Counter(col_vals)
    del counter['']
    freq = counter.most_common(vocab_size-shift)

    return {word: i+shift for i, (word, c) in enumerate(freq)}


def yaml_serializable(val):
    if type(val) is dict:
        val = {k: yaml_serializable(v) for k, v in val.items()}
    elif type(val) in (list, tuple, np.ndarray):
        val = list(map(yaml_serializable, val))
    elif type(val) in (np.float32, np.float64):
        val = float(val)
    elif type(val) in (np.int32, np.int64):
        val = int(val)
    elif type(str):
        pass
    else:
        print("** warning ** unsupported type", type(val))

    return val


def hash_strings(strings, seed, mod, encoded=False):
    if not encoded:
        strings = map(str.encode, strings)

    arr = [hashxx(s, seed=seed) for s in strings]

    return np.mod(arr, mod)


def generate_uid():
    return datetime.now().strftime('%m%d_%H%M')
