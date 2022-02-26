import tensorflow as tf
import numpy as np
import json
from PatchPerPix.models import autoencoder
# from funlib.learn.tensorflow.models import unet, conv_pass, crop
from PatchPerPix.models import unet, conv_pass, crop
from PatchPerPix import util
import sys
import os


def mk_net(**kwargs):

    tf.reset_default_graph()

    input_shape = kwargs['input_shape']
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)
    input_shape = (kwargs['num_channels'],) + input_shape

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32,
                         shape=input_shape,
                         name="raw")

    # create a U-Net
    # raw_batched = tf.reshape(raw, (1, kwargs['num_channels']) + input_shape)
    raw_batched = tf.reshape(raw, (1,) + input_shape)
    model = unet(raw_batched,
                 num_fmaps=kwargs['num_fmaps'],
                 fmap_inc_factors=kwargs['fmap_inc_factors'],
                 fmap_dec_factors=kwargs['fmap_dec_factors'],
                 downsample_factors=kwargs['downsample_factors'],
                 activation=kwargs['activation'],
                 padding=kwargs['padding'],
                 kernel_size=kwargs['kernel_size'],
                 num_repetitions=kwargs['num_repetitions'],
                 upsampling=kwargs['upsampling'])
    print(model)

    num_patch_fmaps = kwargs['autoencoder']['code_units']
    model = conv_pass(
        model,
        kernel_size=1,
        num_fmaps=num_patch_fmaps + 1 + 7,
        padding=kwargs['padding'],
        num_repetitions=1,
        activation=None,
        name="output")
    print(model)

    logits = tf.squeeze(model, axis=0)
    output_shape = logits.get_shape().as_list()[1:]

    logits_code, logits_fgbg, logits_class = tf.split(
        logits, [num_patch_fmaps, 1, 7], 0)

    pred_code = tf.sigmoid(logits_code)

    raw_cropped = crop(raw, [kwargs['num_channels']] +output_shape)

    patchshape_squeezed = tuple(p for p in kwargs['patchshape']
                                if p > 1)
    patchsize = int(np.prod(patchshape_squeezed))
    # placeholder for gt
    gt_affs_shape = [patchsize] + output_shape
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape,
                             name="gt_affs")
    gt_fgbg = tf.placeholder(tf.float32, shape=[1] + output_shape,
                             name="gt_fgbg")
    gt_class = tf.placeholder(tf.int32, shape=[1] + output_shape,
                             name="gt_class")

    anchor = tf.placeholder(tf.float32, shape=[1] + output_shape,
                            name="anchor")

    # get loss
    loss_fgbg, pred_fgbg, _ = \
       util.get_loss(gt_fgbg, logits_fgbg,
                     kwargs['loss'], "fgbg", True)

    logits_class_t = tf.transpose(tf.sigmoid(logits_class), [1, 2, 0])
    gt_class_s = tf.squeeze(gt_class, 0)
    loss_class = tf.losses.sparse_softmax_cross_entropy(gt_class_s, logits_class_t)
    # loss_class = tf.Print(loss_class, [gt_class_s, logits_class_t,
    #                                    tf.math.reduce_min(gt_class_s),
    #                                    tf.math.reduce_max(gt_class_s)],
    #                       message="loss_class",
    #                       first_n=50, summarize=512)
    pred_class = tf.nn.softmax(logits_class, axis=0, name="semantic_class")

    code = tf.transpose(tf.sigmoid(logits_code), [1, 2, 0])
    sample_cnt = 1024
    gt_fgbgTmp = tf.squeeze(gt_fgbg, 0)
    gt_fg_loc = tf.where(gt_fgbgTmp)
    samples_loc = tf.random.uniform(
        (tf.math.minimum(sample_cnt, tf.shape(gt_fg_loc)[0]),),
        minval=0,
        maxval=tf.shape(gt_fg_loc)[0],
        dtype=tf.int32)
    samples_loc = tf.gather(gt_fg_loc, samples_loc)
    code_samples = tf.gather_nd(code, samples_loc)
    print(tf.transpose(gt_affs, [1, 2, 0]))
    print(samples_loc)
    gt_affs_samples = tf.gather_nd(tf.transpose(gt_affs, [1, 2, 0]),
                                   samples_loc)

    ae_config = kwargs['autoencoder']
    ae_config['only_decode'] = True
    ae_config['dummy_in'] = tf.placeholder(
        tf.float32, (None,) + patchshape_squeezed)
    ae_config['is_training'] = True
    ae_config['input_shape_squeezed'] = patchshape_squeezed
    net, sums, _ = autoencoder(code_samples, **ae_config)

    # get loss
    net = tf.reshape(net, (-1, patchsize), name="rs2")
    print(gt_affs_samples, net, gt_affs, gt_affs_shape, samples_loc, gt_fg_loc)

    losspatch, _, _ = \
       util.get_loss(gt_affs_samples, net,
                     kwargs['loss'], "affs", False)
    loss = losspatch + loss_fgbg + loss_class
    loss = tf.Print(loss, [loss, losspatch, loss_fgbg, loss_class], message="losses",
                           first_n=500, summarize=512, name="losses")

    loss_sums = []
    loss_sums.append(tf.summary.scalar('loss_sum', loss))
    loss_sums.append(tf.summary.scalar('loss_affs', losspatch))
    loss_sums.append(tf.summary.scalar('loss_fgbg', loss_fgbg))
    loss_sums.append(tf.summary.scalar('loss_class', loss_class))
    summaries = tf.summary.merge(loss_sums, name="summaries")

    global_step = tf.Variable(0, name='global_step', trainable=False,
                              dtype=tf.int64)

    # optimizer
    learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                name="learning-rate")
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_affs': gt_affs.name,
        'gt_fgbg': gt_fgbg.name,
        'gt_class': gt_class.name,
        'pred_code': pred_code.name,
        'pred_class': pred_class.name,
        'pred_fgbg': pred_fgbg.name,
        #'loss_weights_fgbg': loss_weights_fgbg.name,
        'anchor': anchor.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape[-2:],
        'gt_affs_shape': gt_affs_shape[-2:],
        'output_shape': output_shape[-2:],
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)
