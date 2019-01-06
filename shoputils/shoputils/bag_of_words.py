from konlpy.tag import Okt
import re
from tqdm import tqdm


def tqdm_wrapper(iterator, total):
    if total >= 1000:
        return tqdm(iterator, total=total)

    return iterator


class BagOfWords:
    def __init__(self, coarse_cols, other_cols, backend="twitter"):
        self._col_names = coarse_cols + other_cols
        self._coarse_cols = coarse_cols
        self._tagger = Okt()
        self._number = re.compile("[0-9]")

        self._delimiter = re.compile("[^a-z가-힣9]")

        self._extractors = list(map(re.compile, (
            "([0-9]+x[0-9]+)",  # 120x120
            "([0-9][a-z]{1,4})[^0-9a-z]", # 9cm
            "[^a-z]([a-z][0-9]?\+)", # S3+
            "[^a-z0-9]([a-z][0-9]+)[^0-9a-z]", # M32
            "([0-9]+[가-힣]+)[^가-힣]" # 3게
        )))

        self._keep = set(["Noun", "Adjective", "Alpha", "Adjective", "Foreign", "Modifier"])

    def break_down(self, chunk, decoded=False):
        cols = {}
        L = len(chunk[self._col_names[0]])

        # decoding
        if not decoded:
            for col_name in self._col_names:
                cols[col_name] = (s.decode("utf-8") for s in chunk[col_name])
        else:
            cols = chunk

        # case normalization
        for col_name in self._col_names:
            cols[col_name] = [s.lower() for s in cols[col_name]]

        # coarse extraction
        coarse = []
        for parts in tqdm_wrapper(zip(*(cols[c] for c in self._coarse_cols)), total=L):
            arr = []
            for part in parts:
                arr += self._delimiter.split(part)
            coarse.append(list(filter(bool, arr)))

        # custom extraction
        custom = []
        for parts in tqdm_wrapper(zip(*cols.values()), total=L):
            arr = []
            custom.append(arr)
            for extractor in self._extractors:
                for part in parts:
                    for m in extractor.findall(part):
                        arr.append(m)

        # pos tagging
        pos = []
        for parts in tqdm_wrapper(zip(*cols.values()), total=L):
            arr = []
            pos.append(arr)
            cleaned = (self._number.sub("9", p) for p in parts)
            tagged = (self._tagger.pos(p) for p in cleaned)
            for token, tag in sum(tagged, []):
                if tag in self._keep:
                    arr.append(token)

        # merge
        bags = []
        for coarse_seq, custom_seq, pos_seq in zip(coarse, custom, pos):
            bag = set(coarse_seq)
            bag.update(custom_seq)
            bag.update(pos_seq)
            bags.append(bag)

        return bags

if __name__ == "__main__":
    chunk = {
        "col1": ["후추통-감성쇼핑", "val1"],
        "col2": ["val2", "val2"],
        "col3": ["120x120 11cm S3+ 12일 M32", "[오바쿠(OBAKU)] OBAKU 오바쿠 V149LXCIRB 가죽밴드 여성시계 [V149LXCIRB]"],
        "col4": ["val4", "val4"]
    }

    bow = BagOfWords(coarse_cols=["col1", "col2"], other_cols=["col3", "col4"])
    bags = bow.break_down(chunk, decoded=True)
    print(bags)
