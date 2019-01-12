#!/usr/bin/env python3

import numpy as np
import sys
import pandas as pd
from shoputils.hdf5_io import read_hdf5_sets

key = "train"
col = sys.argv[1]
files = sys.argv[2:]

for filename in files:
    for chunk in read_hdf5_sets(
            set_files=[filename],
            chunk_size=99999999,
            cols=[col],
            key=key):

        if type(chunk[col][0]) in (int, float, np.int32, np.float32, np.float64):
            s = pd.Series(chunk[col])
        else:
            s = pd.Series([v.decode("utf-8") for v in chunk[col]])

        print("\n")
        print(filename)
        print(s.value_counts())
