import numpy as np
import h5py
from skimage import io

inf = h5py.File('/home/maisl/projects/ppp/images/pipeline/patched_selected.hdf',
                'r')
selected = np.array(inf['volumes/pred_affs_patched'])
dst = np.zeros_like(selected)

color_patch = [255, 255, 255]
color_selected = [255,255,255]

patched = selected[2]
print(np.sum(patched > 0.9))

sel = selected[0]

for i in range(3):
    dst[i] = patched * color_patch[i]

mask = sel > 0
for i in range(3):
    dst[i][mask] = sel[mask] * color_selected[i]

dst = dst[0]
io.imsave('/home/maisl/projects/ppp/images/pipeline'
          '/patched.tif',
          np.round(dst).astype(np.uint8))
# io.imsave('/home/maisl/projects/ppp/images/pipeline'
#           '/patched.tif',
#           np.round(np.moveaxis(dst, 0, -1)).astype(np.uint8))


#
# inf = h5py.File('/home/maisl/projects/ppp/images/pipeline/D08_instances'
#                 '.hdf', 'r')
# src = np.squeeze(np.array(inf['vote_instances']))
# for i in range(src.shape[0]):
#     mask = src[i] > 0
#     if np.sum(mask) > 0:
#         io.imsave('/home/maisl/projects/ppp/images/pipeline/D08_inst_%i.png' % (
#             i+1), (mask * 255).astype(np.uint8))
