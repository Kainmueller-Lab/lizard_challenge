import numpy as np
import zarr
import os
import sys

ind = sys.argv[1]
fls = os.listdir(ind)

tmp = zarr.open(os.path.join(ind, fls[0]), 'r')['volumes/raw']

shape = list(tmp.shape[-2:])

out_raw = np.zeros([len(fls)] + shape + [3], dtype=tmp.dtype)
out_label = np.zeros([len(fls)] + shape + [2], dtype=np.uint16)
for idx, fl in enumerate(fls):
    fl = zarr.open(os.path.join(ind, fl), 'r')
    tmp = fl['volumes/raw']
    out_raw[idx] = np.transpose(tmp, (1, 2, 0))
    tmp = fl['volumes/gt_instances']
    out_label[idx, ..., 0] = tmp
    tmp = fl['volumes/gt_class']
    out_label[idx, ..., 1] = tmp

np.save("val_raw.npy", out_raw)
np.save("val_labels.npy", out_label)
