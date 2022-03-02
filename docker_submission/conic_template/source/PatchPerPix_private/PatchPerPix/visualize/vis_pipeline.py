import numpy as np
import h5py
import zarr
import argparse
import skimage.io
import skimage.transform
import logging
import scipy.special
import scipy.ndimage
import os
import sys
import mahotas

logger = logging.getLogger(__name__)

def resize_indices(input_shape: tuple, output_shape: tuple) -> np.ndarray:
  """Produces indices for resampling input shape into output shape."""
  if len(input_shape) != len(output_shape):
    raise ValueError('incompatible shape dimensions')

  indices = np.indices(output_shape, dtype=np.float32)
  for d in range(len(output_shape)):
    s = input_shape[d] / output_shape[d]
    indices[d] *= np.float32(s)
    indices[d] += np.float32(s / 2 - 0.5)  # input vs output pixel center displacement

  return indices


def resize(im: np.ndarray, shape: tuple, optimize_map: bool=True, **kwargs) -> np.ndarray:
  """Somehow "proper" implementation, instead of buggy scipy.ndimage.zoom"""
  if len(im.shape) != len(shape):
    raise ValueError('incompatible shape dimensions')

  if 'output' in kwargs:
    output = kwargs['output']
    del kwargs['output']
  else:
    output = np.empty(shape, dtype=im.dtype)

  if im.shape == shape:
    np.copyto(output, im)
    return output

  # optimize doing the interpolation for N-1 dimensions
  if optimize_map and im.shape[0] == shape[0]:
    indices = resize_indices(im.shape[1:], shape[1:])
    for i in range(shape[0]):
      scipy.ndimage.map_coordinates(im[i], indices, output=output[i], **kwargs)
    return output

  return scipy.ndimage.map_coordinates(im, resize_indices(im.shape, shape), output=output, **kwargs)


def reshape_affinities(in_array, patchshape):
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    py, px = patchshape
    logger.debug("%s %s %s %s", in_array.dtype,
                 np.min(in_array), np.max(in_array), in_array.shape)
    patched = np.zeros(
        (3, in_array.shape[1] * py, in_array.shape[2] * px),
        dtype=in_array.dtype)
    logger.info('transforming affs shape %s into %s',
                in_array.shape, patched.shape)
    for y in range(in_array.shape[1]):
        logger.debug('processing %i / %i', y, in_array.shape[1])
        for x in range(in_array.shape[2]):
            p = np.copy(in_array[:, y, x])
            p.shape = (25, 25)
            p = p[7:18, 7:18]
            p[0, :] = 1
            p[:, 0] = 1
            p[0, :] = 1
            p[:, 0] = 1
            p[0, :] = 1
            p[:, 0] = 1
            patched[0,
                    y * py:(y + 1) * py,
                    x * px:(x + 1) * px] = p
            patched[1,
                    y * py:(y + 1) * py,
                    x * px:(x + 1) * px] = p
            patched[2,
                    y * py:(y + 1) * py,
                    x * px:(x + 1) * px] = p

    return patched



def visualize_patches(
    affinities,
    patchshape,
    in_key,
    out_file=None,
    threshold=None,
    sigmoid=False,
    store_int=False
):
    """Visualize sequential affinities by reshaping to patchshape and
    separating them visually. Affinities can be either filename or numpy array.
    """
    if affinities.endswith('.zarr'):
        inf = zarr.open(affinities, mode='r')
    elif affinities.endswith('.hdf'):
        inf = h5py.File(affinities, 'r')
    else:
        raise NotImplementedError

    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    labels = np.squeeze(np.array(inf[in_key]))
    logger.info("file type/min/max %s %s %s", labels.dtype,
                np.min(labels), np.max(labels))

    try:
        raw = np.squeeze(np.array(inf["volumes/raw_bf"]))
    except Exception as e:
        print(e)
        print("no raw data in file")
        raw = None

    # CHW vs HWC
    py, px = patchshape
    patchsize = py*px
    if labels.shape[0] != patchsize and labels.shape[-1] == patchsize:
        labels = np.ascontiguousarray(np.moveaxis(labels, -1, 0))
    # with/without sigmoid applied?
    if sigmoid:
        labels = scipy.special.expit(labels)
        logger.info("after sigmoid: file type/min/max %s %s %s",
                    labels.dtype, np.min(labels), np.max(labels))

    labels = np.clip(labels, 0.3, 1)
    labels = (labels - 0.3)/(1.0 - 0.3)

    # reshape sequential affinities to patchshape
    fn = os.path.splitext(out_file)[0] + "patched.hdf"
    try:
        with h5py.File(fn, 'r') as f:
            patched = np.array(f['volumes/patched'])
    except:
        patched = reshape_affinities(labels, patchshape)
        with h5py.File(fn, 'w') as f:
            f.create_dataset(
                'volumes/patched',
                data=patched,
                compression='gzip')

    if threshold is not None:
        patched[patched < threshold] = 0

    return patched, labels, raw


def set_rgb(arr, y, x, r, g, b):
    arr[0, y, x] = r
    arr[1, y, x] = g
    arr[2, y, x] = b


def set_pixel_box(patched, py, px, ps, r, g, b):
    for y in range(0, ps):
        ty = py * ps + y
        for ddx in [0, ps]:
            tx = px * ps + ddx
            for dx in range(-1, 2):
                txt = tx + dx
                for dy in range(-1, 2):
                    tyt = ty + dy
                    set_rgb(patched, tyt, txt, r, g, b)

    for x in range(0, ps):
        tx = px * ps + x
        for ddy in [0, ps]:
            ty = py * ps + ddy
            for dx in range(-1, 2):
                txt = tx + dx
                for dy in range(-1, 2):
                    tyt = ty + dy
                    set_rgb(patched, tyt, txt, r, g, b)


def set_neighbor_box(patched, py, px, ps, r, g, b):
    x = (px-ps//2)*ps
    for y in range((py-ps//2)*ps, (py+ps//2+1)*ps):
        set_rgb(patched, y, x, r, g, b)
        set_rgb(patched, y, x-1, r, g, b)

    x = (px+ps//2+1)*ps
    for y in range((py-ps//2)*ps, (py+ps//2+1)*ps):
        set_rgb(patched, y, x, r, g, b)
        set_rgb(patched, y, x+1, r, g, b)

    y = (py-ps//2)*ps
    for x in range((px-ps//2)*ps, (px+ps//2+1)*ps):
        set_rgb(patched, y, x, r, g, b)
        set_rgb(patched, y-1, x, r, g, b)

    y = (py+ps//2+1)*ps
    for x in range((px-ps//2)*ps, (px+ps//2+1)*ps):
        set_rgb(patched, y, x, r, g, b)
        set_rgb(patched, y+1, x, r, g, b)


def set_point_in_patches(patched, py, px, ps, r, g, b):
    for y in range(-ps//2, ps//2+1):
        for x in range(-ps//2, ps//2+1):
            ty = (py - y) * ps + y + ps//2
            tx = (px - x) * ps + x + ps//2

            typ = ty+1
            tym = ty-1
            txp = tx+1
            txm = tx-1
            # t1yp1 = min(t1y+1, (py + y + 1) * ps -1)
            # t1ym1 = max(t1y-1, (px + x - 1) * ps)
            # t1xp1 = min(t1x+1, (py + y + 1) * ps -1)
            # t1xm1 = max(t1x-1, (px + x - 1) * ps)

            set_rgb(patched, ty, tx, r, g, b)
            set_rgb(patched, typ, tx, r, g, b)
            set_rgb(patched, tym, tx, r, g, b)
            set_rgb(patched, ty, txp, r, g, b)
            set_rgb(patched, ty, txm, r, g, b)
            set_rgb(patched, typ, txp, r, g, b)
            set_rgb(patched, tym, txm, r, g, b)
            set_rgb(patched, typ, txm, r, g, b)
            set_rgb(patched, tym, txp, r, g, b)


def crop_patched(patched, points, ps, affinities, scores, raw):

    points_x = points[::2]
    points_y = points[1::2]

    min_x = min(points_x) - ps//2 - ps//5
    max_x = max(points_x) + ps//2 + ps//5
    min_y = min(points_y) - ps//2 - ps//5
    max_y = max(points_y) + ps//2 + ps//5

    # points = points[:2]

    rgbs = [[1, 0, 0], [0, 1, 1], [1, 1, 0],
            [1, 1, 0]]


    instances_tmp = np.copy(scores)
    scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
    for i in range(0, len(points), 2):
        px = points[i]
        py = points[i+1]
        scores[py, px] += 0.1
    print(scores.shape)
    scores = scipy.ndimage.zoom(scores, ps, order=0)
    scores = np.stack([scores, scores, scores])


    instances = None
    seeds = np.zeros(instances_tmp.shape)
    for i in range(0, len(points), 2):
        px = points[i]
        py = points[i+1]
        seeds[py, px] = i
        instances_tmp[py, px] = 1
    instances_tmp = scipy.ndimage.zoom(instances_tmp, ps, order=0)
    seeds = scipy.ndimage.zoom(seeds, ps, order=0)
    instances_tmp = instances_tmp[(ps*min_y):(ps*max_y),
                                  (ps*min_x):(ps*max_x)]
    seeds = seeds[(ps*min_y):(ps*max_y),
                  (ps*min_x):(ps*max_x)].astype(np.uint8)

    instances_tmp2 = mahotas.cwatershed(instances_tmp, seeds)
    print(np.unique(instances_tmp2))
    l = np.unique(instances_tmp2)
    instances = np.zeros((3,) + tuple(instances_tmp2.shape))
    instances[0][instances_tmp2 == l[0]] = 1
    instances[1][instances_tmp2 == l[0]] = 1
    instances[0][instances_tmp2 == l[1]] = 1
    instances[2][instances_tmp2 == l[1]] = 1
    instances[0][instances_tmp < 0.5] = 0
    instances[1][instances_tmp < 0.5] = 0
    instances[2][instances_tmp < 0.5] = 0

    for i in range(0, len(points), 2):
        px = points[i]
        py = points[i+1]
        set_neighbor_box(patched, py, px, ps,
                         rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])
        set_pixel_box(patched, py, px, ps,
                      rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])
        set_point_in_patches(patched, py, px, ps,
                             rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])
        if scores is not None:
            set_pixel_box(scores, py, px, ps,
                      rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])


    if affinities is None:
        patched_zoom = None
    else:
        patched_zoom = np.copy(scores)
        for i in range(0, len(points), 2):
            px = points[i]
            py = points[i+1]
            set_neighbor_box(patched_zoom, py, px, ps,
                             rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])
            set_neighbor_box(scores, py, px, ps,
                             rgbs[i//2][0], rgbs[i//2][1], rgbs[i//2][2])
        for i in range(0, len(points), 2):
            px = points[i]
            py = points[i+1]
            p = affinities[:, py, px]
            p.shape = (25, 25)
            p = np.copy(p[6:19, 6:19]).astype(np.float32)
            p = resize(p, tuple([s*ps for s in p.shape]), order=0)
            p = p[...,ps:-ps, ps:-ps]
            # p = scipy.ndimage.zoom(p, ps, order=0)
            patched_zoom[:,
                         ps*(py-ps//2):ps*(py+ps//2+1),
                         ps*(px-ps//2):ps*(px+ps//2+1)] = 0
        for i in range(0, len(points), 2):
            px = points[i]
            py = points[i+1]
            p = affinities[:, py, px]
            p.shape = (25, 25)
            p = np.copy(p[6:19, 6:19]).astype(np.float32)
            p = resize(p, tuple([s*ps for s in p.shape]), order=0)
            p = p[...,ps:-ps, ps:-ps]
            # p = scipy.ndimage.zoom(p, ps, order=0)
            if i == 0:
                patched_zoom[0,
                             ps*(py-ps//2):ps*(py+ps//2+1),
                             ps*(px-ps//2):ps*(px+ps//2+1)] = p*0.7
            elif i == 2:
                # patched_zoom[1,
                #              ps*(py-ps//2):ps*(py+ps//2+1),
                #              ps*(px-ps//2):ps*(px+ps//2+1)] = p*0.5
                patched_zoom[2,
                             ps*(py-ps//2):ps*(py+ps//2+1),
                             ps*(px-ps//2):ps*(px+ps//2+1)] = p*0.7

            elif i == 4:
                for yz, yp in zip(
                        range(ps*(py-ps//2),ps*(py+ps//2+1)),
                        range(p.shape[0])):
                    for xz, xp in zip(
                            range(ps*(px-ps//2),ps*(px+ps//2+1)),
                            range(p.shape[1])):
                        if patched_zoom[0, yz, xz] <= 0.1:
                            patched_zoom[0, yz, xz] = p[yp, xp] * 0.7
                patched_zoom[1,
                             ps*(py-ps//2):ps*(py+ps//2+1),
                             ps*(px-ps//2):ps*(px+ps//2+1)] = p*0.7

        patched_zoom = patched_zoom[...,
                      (ps*min_y):(ps*max_y),
                      (ps*min_x):(ps*max_x)]

    patched = patched[...,
                      (ps*min_y):(ps*max_y),
                      (ps*min_x):(ps*max_x)]
    tmp = np.copy(patched)
    tmpmn = np.min(tmp)
    tmp[tmp==1] = 0
    tmpmx = np.max(tmp)

    tmp = np.copy(patched)
    patched = (patched-tmpmn)/(tmpmx-tmpmn)
    patched[tmp==1] = 1

    scores = scores[...,
                    (ps*min_y):(ps*max_y),
                    (ps*min_x):(ps*max_x)]

    raw = scipy.ndimage.zoom(raw, ps, order=0)
    raw = raw[(ps*min_y):(ps*max_y),
              (ps*min_x):(ps*max_x)]

    return patched, scores, patched_zoom, instances, raw

def main():
    logging.basicConfig(level=20)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--in-key', type=str, dest='in_key',
                        help='input key', required=True)
    parser.add_argument('--patchshape', type=int,
                        help='patchshape', nargs='+', required=True)
    parser.add_argument('--out-file', type=str, default=None,
                        dest='out_file', help='output file')
    parser.add_argument('--out-key', type=str, default=None,
                        dest='out_key', help='output key')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Value to threshold predictions')
    parser.add_argument("--sigmoid", action="store_true",
                        help='apply sigmoid')
    parser.add_argument("--scores", type=str, default=None,
                        help='scores file')
    parser.add_argument("--store_int", action="store_true",
                        help='store patched array as uint8')
    parser.add_argument("-p", "--points", default=[], nargs="+",
                        help="restrict output to bb of these points")
    args = parser.parse_args()
    if args.scores is not None:
        scores = skimage.io.imread(args.scores)
    else:
        scores = None

    patched, affinities, raw = visualize_patches(
        args.in_file, args.patchshape,
        args.in_key,
        threshold=args.threshold,
        out_file=args.out_file,
        sigmoid=args.sigmoid,
        store_int=args.store_int)

    print(patched.shape)

    points = [int(p) for p in args.points]
    patched, scores, zoomed, instances, raw = crop_patched(
        patched, points, args.patchshape[-1], affinities, scores,
        raw)
    print(patched.shape)

    if args.out_key is None:
        args.out_key = args.in_key + '_patched'

    if patched.dtype == np.float16:
        patched = patched.astype(np.float32)

    with h5py.File(args.out_file, 'w') as outf:
        outf.create_dataset(
            args.out_key,
            data=patched,
            compression='gzip'
        )
    skimage.io.imsave(
        os.path.splitext(args.out_file)[0] + ".png",
        (np.transpose(patched, [1, 2, 0])*255).astype(np.uint8))
    if scores is not None:
        skimage.io.imsave(
            os.path.splitext(args.out_file)[0] + "scores.png",
            (np.transpose(scores, [1, 2, 0])*255).astype(np.uint8))

    skimage.io.imsave(
        os.path.splitext(args.out_file)[0] + "zoomed.png",
        (np.transpose(zoomed, [1, 2, 0])*255).astype(np.uint8))

    skimage.io.imsave(
        os.path.splitext(args.out_file)[0] + "instances.png",
        (np.transpose(instances, [1, 2, 0])*255).astype(np.uint8))

    skimage.io.imsave(
        os.path.splitext(args.out_file)[0] + "raw.png",
        (raw*255).astype(np.uint8))

if __name__ == "__main__":
    main()
