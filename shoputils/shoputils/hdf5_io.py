import h5py
import numpy as np

def indices_to_mask(indices, size):
    # do slicing with booleans because it is much faster
    # https://github.com/h5py/h5py/issues/368

    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def read_hdf5_sets(
        set_files,
        chunk_size,
        cols,
        min_month=1,
        max_month=12,
        min_year=0,
        max_year=3000,
        key="train",
        loc=None):

    moving_avg = 1.05

    if loc is not None:
        from_file, start_row = loc
        set_files = set_files[set_files.index(from_file):]
    else:
        start_row = 0

    for set_file in set_files:
        print("reading", set_file)
        if start_row is not None:
            i = start_row
            start_row = None
        else:
            i = 0
        with h5py.File(set_file, "r") as f:
            grp = f[key]
            col_data = {col: grp[col] for col in cols}  # perhaps unecessary? depending on how caching works under the hood
            n_rows = len(col_data[cols[0]])
            while i < n_rows:
                # preemptively read a little bit more to make sure we get enough
                to_read = max(1, int(1.05 * chunk_size / moving_avg))
                chunk = {
                    col: col_data[col][i:i+to_read] for col in cols
                }

                # filter by month
                if min_month != 1 or max_month != 12 or min_year != 0 or max_year != 3000:
                    dates = grp["updttm"][i:i+to_read]
                    month = np.array([int(d[4:6]) for d in dates])
                    years = np.array([int(d[:4]) for d in dates])
                    filt = np.where((years >= min_year) & (years <= max_year) &
                                    (month >= min_month) & (month <= max_month))[0]
                    actual_size = len(filt)
                    if actual_size == 0:
                        i += to_read
                        continue
                    chunk = {
                        col: vals[filt] for col, vals in chunk.items()
                    }
                    moving_avg = 0.9 * moving_avg + 0.1 * (chunk_size/actual_size)

                if loc is None:
                    yield chunk
                else:
                    yield chunk, (set_file, i)

                i += to_read
