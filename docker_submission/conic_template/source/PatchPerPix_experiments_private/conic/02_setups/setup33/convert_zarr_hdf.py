import zarr
import numpy as np
import h5py
import sys
import os

zfn = sys.argv[1]
hfn = os.path.splitext(os.path.basename(zfn))[0] + "_.hdf"

if os.path.splitext(zfn)[-1] == ".hdf":
    zfl = h5py.File(zfn, 'r')
else:
    zfl = zarr.open(zfn, 'r')

with h5py.File(hfn, 'w') as hfl:
    for k in zfl['volumes'].keys():
        if zfl['volumes'][k].dtype in(np.float64, np.float16, 'float64', 'float16'):
            dtype = np.float32
        elif zfl['volumes'][k].dtype == np.int64 or \
           zfl['volumes'][k].dtype == np.int32:
            dtype = np.uint16
        else:
            dtype = zfl['volumes'][k].dtype
            print(k, dtype)
        hfl.create_dataset(k,
                           data=np.array(zfl['volumes'][k]).astype(dtype),
                           compression='gzip')

if os.path.splitext(zfn)[-1] == ".hdf":
    zfl.close()
