#!/usr/bin/env python3

import numpy as np
import sys
import pandas as pd
from collections import Counter
from shoputils.hdf5_io import read_hdf5_sets

files = sys.argv[1:]
date_col = "updttm"
key = "train"

counter = Counter()

for filename in files:
    for chunk in read_hdf5_sets(
            set_files=[filename],
            chunk_size=99999999,
            cols=[date_col],
            key=key):

        decoded = (d.decode() for d in chunk[date_col])
        times = np.array([13 * int(d[0:4]) + int(d[4:6]) for d in decoded])
        counter.update(times)

total = 0
vals = []
for key, c in counter.items():
    month = key % 13
    year = key / 13
    vals.append("%i %2i: %i" % (year, month, c))
    total += c

print("total:", total)

vals.sort()
for s in vals:
    print(s)
