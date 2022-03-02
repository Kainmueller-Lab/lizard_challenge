import sys
import os
import numpy as np
import gunpowder as gp
import json
import logging
from datetime import datetime
import zarr
import h5py

import tensorflow as tf

def predict(**kwargs):
    name = kwargs['name']
    samples = kwargs['samples']

    if kwargs['output_format'] != "zarr" and kwargs['output_format'] != "hdf":
        raise NotImplementedError("Please use zarr or hdf as prediction output")

    session = tf.Session()
    meta_graph_file = kwargs['checkpoint'] + '.meta'
    input_map = None
    # if self.is_training is not None:
    #     input_map = {}
    #     if self.do_test_time_reps:
    #         input_map[self.is_training] = tf.constant(True)
    #     else:
    #         input_map[self.is_training] = tf.constant(False)
    saver = tf.train.import_meta_graph(
        meta_graph_file,
        input_map=input_map,
        clear_devices=True)

    saver.restore(session, kwargs['checkpoint'])


    with open(os.path.join(kwargs['input_folder'],
                           name + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['input_folder'],
                           name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape']) * voxel_size
    context = (input_shape_world - output_shape_world) // 2

    raw_key = kwargs.get('raw_key', 'volumes/raw')
    code_key = kwargs.get('code_key', 'volumes/pred_code')
    fg_key = kwargs.get('fg_key', 'volumes/pred_fgbg')
    class_key = kwargs.get('class_key', 'volumes/pred_class')

    print(samples)
    npy_source = False
    if len(samples) > 0 and samples[0].endswith("npy"):
        npy_source = True
        print("loading npy", samples)
        samples = samples[0]
        samples_data = np.load(samples)
        samples_data = np.ascontiguousarray(np.moveaxis(samples_data, -1, 1))
        print(samples_data.shape)
        samples = ["patch_{}".format(idx) for idx in range(samples_data.shape[0])]

    for idx, sample in enumerate(samples):
        # if idx > 10:
        #     break
        print("predicting {}/{}: {}, tta? {}".format(idx, len(samples), sample,
                                                     kwargs.get('test_time_aug')))
        if npy_source:
            raw = np.array(samples_data[idx])
        elif kwargs['input_format'] == "hdf":
            with h5py.File(os.path.join(kwargs['data_folder'],
                                        sample + ".hdf"), 'r') as f:
                raw = np.array(f[raw_key])
        elif kwargs['input_format'] == "zarr":
            f = zarr.open(os.path.join(kwargs['data_folder'],
                                       sample + ".zarr"), 'r')
            raw = np.array(f[raw_key])
        else:
            raise NotImplementedError("predict node for %s not implemented yet",
                                      kwargs['input_format'])
        shape = raw.shape[-2:]
        print(shape)


        # crop = []
        # print(shape, net_config['output_shape'])
        # for d in range(-2, 0):
        #     if shape[d] < net_config['output_shape'][d]:
        #         crop.append((net_config['output_shape'][d]-shape[d])//2)
        #     else:
        #         crop.append(0)
        # print("cropping", crop)
        # print(context)

        raw = raw/255.0
        rawS = np.stack([raw]*kwargs['batch_size'], axis=0)

        outputs = [
            net_names['pred_code'],
            net_names['pred_fgbg'],
            net_names['pred_class'],
        ]

        if kwargs.get('test_time_aug'):
            print("normal")
        input_data = {net_names['raw']: rawS}
        output_data = session.run(
            {t: t for t in outputs},
            feed_dict=input_data)

        pred_codeT = output_data[net_names['pred_code']][0]
        pred_fgbgT = output_data[net_names['pred_fgbg']][0]
        pred_classT = output_data[net_names['pred_class']][0]


        cu = kwargs['code_units']
        if kwargs.get('test_time_aug'):
            cu = kwargs['code_units'] * 4
            pred_codes = [pred_codeT]
            rawT = np.flip(raw, 1)
            rawS = np.stack([rawT]*kwargs['batch_size'], axis=0)

            print("flippedlr")
            input_data = {net_names['raw']: rawS}
            output_data = session.run(
                {t: t for t in outputs},
                feed_dict=input_data)

            pred_fgbgT += np.flip(output_data[net_names['pred_fgbg']][0], 1)
            pred_classT += np.flip(output_data[net_names['pred_class']][0], 1)
            pred_codes.append(np.flip(output_data[net_names['pred_code']][0], 1))


            rawT = np.flip(raw, 2)
            rawS = np.stack([rawT]*kwargs['batch_size'], axis=0)

            print("flippedud")
            input_data = {net_names['raw']: rawS}
            output_data = session.run(
                {t: t for t in outputs},
                feed_dict=input_data)

            pred_fgbgT += np.flip(output_data[net_names['pred_fgbg']][0], 2)
            pred_classT += np.flip(output_data[net_names['pred_class']][0], 2)
            pred_codes.append(np.flip(output_data[net_names['pred_code']][0], 2))



            rawT = np.rot90(raw, 1, axes=(1,2))
            rawS = np.stack([rawT]*kwargs['batch_size'], axis=0)

            print("rot90")
            input_data = {net_names['raw']: rawS}
            output_data = session.run(
                {t: t for t in outputs},
                feed_dict=input_data)

            pred_fgbgT += np.rot90(output_data[net_names['pred_fgbg']][0], -1, axes=(1,2))
            pred_classT += np.rot90(output_data[net_names['pred_class']][0], -1, axes=(1,2))
            pred_codes.append(np.rot90(output_data[net_names['pred_code']][0], -1, axes=(1,2)))

            pred_fgbgT /= 4
            pred_classT /= 4
            pred_codeT = np.concatenate(pred_codes, axis=0)
            # pred_codeT = np.fliplr(output_data[net_names['pred_code']][0])
            # for h in range(pred_codeT.shape[-2]):
            #     for w in range(pred_codeT.shape[-1]):
            #         code = pred_codeT[:, h, w]
            #         code = np.reshape(code, (25, 25))
            #         code = np.fliplr(code)
            #         code = np.reshape(code, (625,))
            #         #

        if kwargs['output_format'] == "zarr":
            # open zarr file
            zf = zarr.open(os.path.join(kwargs['output_folder'],
                                        sample + '.zarr'), mode='w')

            pred_code = zf.create(code_key,
                                  shape=[cu] + list(shape),
                                  chunks=[kwargs['code_units']] + list(shape),
                                  dtype=np.float32)
            zf[code_key].attrs['offset'] = [0, 0]
            zf[code_key].attrs['resolution'] = kwargs['voxel_size']

            pred_fgbg =  zf.create(fg_key,
                                   shape=[1] + list(shape),
                                   chunks=[1] + list(shape),
                                   dtype=np.float32)
            zf[fg_key].attrs['offset'] = [0, 0]
            zf[fg_key].attrs['resolution'] = kwargs['voxel_size']

            pred_class = zf.create(class_key,
                                   shape=[7] + list(shape),
                                   chunks=[7] + list(shape),
                                   dtype=np.float32)
            zf[class_key].attrs['offset'] = [0, 0]
            zf[class_key].attrs['resolution'] = kwargs['voxel_size']

            raw_out = zf.create(raw_key,
                                shape=[3] + list(shape),
                                chunks=[3] + list(shape),
                                dtype=np.float32)
            zf[raw_key].attrs['offset'] = [0, 0]
            zf[raw_key].attrs['resolution'] = kwargs['voxel_size']

            pred_code[:] = pred_codeT
            pred_fgbg[:] = pred_fgbgT
            pred_class[:] = pred_classT
            raw_out[:] = rawS[0]
        else:
            with h5py.File(os.path.join(kwargs['output_folder'],
                                        sample + '.hdf'), mode='w') as hf:
                hf.create_dataset(code_key,
                                  data=pred_codeT,
                                  # chunks=[kwargs['code_units']] + list(shape),
                                  dtype=np.float32)
                hf[code_key].attrs['offset'] = [0, 0]
                hf[code_key].attrs['resolution'] = kwargs['voxel_size']

                hf.create_dataset(fg_key,
                                  data=pred_fgbgT,
                                  # chunks=[1] + list(shape),
                                  dtype=np.float32)
                hf[fg_key].attrs['offset'] = [0, 0]
                hf[fg_key].attrs['resolution'] = kwargs['voxel_size']

                hf.create_dataset(class_key,
                                  data=pred_classT,
                                  # chunks=[7] + list(shape),
                                  dtype=np.float32)
                hf[class_key].attrs['offset'] = [0, 0]
                hf[class_key].attrs['resolution'] = kwargs['voxel_size']

                hf.create_dataset(raw_key,
                                  data=rawS[0],
                                  # chunks=[3] + list(shape),
                                  dtype=np.float32)
                hf[raw_key].attrs['offset'] = [0, 0]
                hf[raw_key].attrs['resolution'] = kwargs['voxel_size']
