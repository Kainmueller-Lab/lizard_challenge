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
    input_shape = (kwargs['batch_size'], kwargs['num_channels'],) + input_shape

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32,
                         shape=input_shape,
                         name="raw")

    # create a U-Net
    # raw_batched = tf.reshape(raw, (1, kwargs['num_channels']) + input_shape)
    # raw_batched = tf.reshape(raw, (kwargs['batch_size'],) + input_shape)

    initializer = tf.compat.v1.initializers.he_normal()
    # network_type = "split" # "single"
    network_type = "single"
    if network_type == "single":
        model = unet(raw,
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
        logits = conv_pass(
            model,
            kernel_size=1,
            num_fmaps=num_patch_fmaps + 1 + 7,
            padding=kwargs['padding'],
            num_repetitions=1,
            activation=None,
            name="output")
        print(logits)
        # logits = tf.squeeze(model, axis=0)
        output_shape = logits.get_shape().as_list()[-2:]

        logits_code, logits_fgbg, logits_class = tf.split(
            logits, [num_patch_fmaps, 1, 7], 1)

    elif network_type == "split":
        with tf.variable_scope("instance",
                               initializer=initializer):
            model = unet(raw,
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
            logits = conv_pass(
                model,
                kernel_size=1,
                num_fmaps=num_patch_fmaps + 1,
                padding=kwargs['padding'],
                num_repetitions=1,
                activation=None,
                name="output")
            print(logits)

        with tf.variable_scope("semantic",
                               initializer=initializer):
            model2 = unet(raw,
                         num_fmaps=kwargs['num_fmaps']//4,
                         fmap_inc_factors=kwargs['fmap_inc_factors'],
                         fmap_dec_factors=kwargs['fmap_inc_factors'],
                         downsample_factors=kwargs['downsample_factors'],
                         activation=kwargs['activation'],
                         padding=kwargs['padding'],
                         kernel_size=kwargs['kernel_size'],
                         num_repetitions=kwargs['num_repetitions'],
                         upsampling=kwargs['upsampling'])
            print(model2)

            logits_class = conv_pass(
                model,
                kernel_size=1,
                num_fmaps=7,
                padding=kwargs['padding'],
                num_repetitions=1,
                activation=None,
                name="output")
            print(logits_class)

        # logits = tf.squeeze(model, axis=0)
        output_shape = logits.get_shape().as_list()[-2:]

        logits_code, logits_fgbg = tf.split(
            logits, [num_patch_fmaps, 1], 1)

    else:
        raise RuntimeError("invalid network type")

    pred_code = tf.sigmoid(logits_code)

    raw_cropped = crop(raw, [kwargs['batch_size'],
                             kwargs['num_channels']] + output_shape)

    patchshape_squeezed = tuple(p for p in kwargs['patchshape']
                                if p > 1)
    patchsize = int(np.prod(patchshape_squeezed))
    # placeholder for gt
    gt_affs_shape = [kwargs['batch_size'], patchsize] + output_shape
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape,
                             name="gt_affs")
    gt_fgbg = tf.placeholder(tf.float32, shape=[kwargs['batch_size'], 1] + output_shape,
                             name="gt_fgbg")
    gt_class = tf.placeholder(tf.int32, shape=[kwargs['batch_size'], 1] + output_shape,
                             name="gt_class")

    anchor = tf.placeholder(tf.float32, shape=[kwargs['batch_size'], 1] + output_shape,
                            name="anchor")

    # get loss
    loss_fgbg, pred_fgbg, _ = \
       util.get_loss(gt_fgbg, logits_fgbg,
                     kwargs['loss'], "fgbg", True)

    logits_class_t = tf.transpose(tf.sigmoid(logits_class), [0, 2, 3, 1])
    gt_class_s = tf.squeeze(gt_class, 1)
    pred_class = tf.nn.softmax(logits_class, axis=1, name="semantic_class")

    num_classes = 7
    if kwargs.get('class_loss') == "balanced_ce":
        # global_ratio = tf.constant([1/7]*7)
        global_ratio = tf.constant([0.84476037,
                                    0.00152521, 0.0979013, 0.01781748,
                                    0.00550755, 0.00104388, 0.03144421])

        global_ratio_normed = tf.math.reduce_max(global_ratio)/global_ratio
        global_ratio_pix = global_ratio*kwargs['batch_size']*256*256
        gt_labels_one_hot = tf.one_hot(gt_class_s, num_classes, axis=0)
        class_cnt = tf.reshape(gt_labels_one_hot, (num_classes, -1))
        class_cnt = tf.math.reduce_sum(class_cnt, axis=-1)

        weight = tf.math.divide(global_ratio_pix, class_cnt) * global_ratio_normed
        weight = tf.gather(weight, gt_class_s)

        print("balanced ce loss", gt_labels_one_hot, weight, gt_class_s, logits_class_t, global_ratio, global_ratio_normed, global_ratio_pix, class_cnt)
        loss_class = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_s,
                                                            logits=logits_class_t,
                                                            weights=weight)
    elif kwargs.get('class_loss') == "balanced_focal":
        # global_ratio = tf.constant([1/7]*7)
        global_ratio = tf.constant([0.84476037,
                                    0.00152521, 0.0979013, 0.01781748,
                                    0.00550755, 0.00104388, 0.03144421])

        global_ratio_normed = tf.math.reduce_max(global_ratio)/global_ratio
        global_ratio_pix = global_ratio*kwargs['batch_size']*256*256
        gt_labels_one_hot = tf.one_hot(gt_class_s, num_classes, axis=0)
        class_cnt = tf.reshape(gt_labels_one_hot, (num_classes, -1))
        class_cnt = tf.math.reduce_sum(class_cnt, axis=-1)

        weightT = tf.math.divide(global_ratio_pix, class_cnt) * global_ratio_normed
        weightT = tf.Print(weightT, [weightT], message="weight",
                           first_n=50, summarize=512, name="weight")
        weight = tf.gather(weightT, gt_class_s)

        weight2 = 1.0 - tf.reduce_sum(
            tf.multiply(
                tf.transpose(gt_labels_one_hot, [1, 0, 2, 3]),
                pred_class),
            axis=1)
        weight2 = weight2*weight2


        weight = weight * weight2



        print("balanced focal loss", gt_labels_one_hot, weight, gt_class_s, logits_class_t, global_ratio, global_ratio_normed, global_ratio_pix, class_cnt)
        loss_class = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_s,
                                                            logits=logits_class_t,
                                                            weights=weight)
    elif kwargs.get('class_loss') == "moving_focal":
        print("moving focal loss")
        running_conf = tf.Variable(tf.ones(num_classes)/num_classes)
        # gt_labels_one_hot = tf.one_hot(gt_class_s, num_classes, axis=1)
        # probs_avg = tf.math.reduce_mean(gt_labels_one_hot, axis=0)
        # probs_avg = tf.reshape(probs_avg, (num_classes, -1))
        # probs_avg = tf.math.reduce_mean(probs_avg, axis=-1)

        gt_labels_one_hot = tf.one_hot(gt_class_s, num_classes, axis=0)
        probs_avg = tf.reshape(gt_labels_one_hot, (num_classes, -1))
        probs_avg = tf.math.reduce_mean(probs_avg, axis=-1)

        # updating the new records: copy the value
        eps = 1e-8

        running_conf = tf.assign(
            running_conf,
            tf.where(tf.math.logical_and(tf.math.greater(probs_avg, eps),
                                         tf.equal(running_conf, 1.0/num_classes)),
                     probs_avg, running_conf))

        momentum = 0.99
        focal_p = 3
        running_conf = tf.assign(
            running_conf,
            running_conf*momentum + (1 - momentum) * probs_avg)

        weight = tf.math.pow((1 - tf.nn.relu(running_conf)), focal_p)
        weight = tf.Print(weight, [weight, running_conf], message="weight",
                          first_n=50, summarize=512, name="weight")
        weight = tf.gather(weight, gt_class_s)

        print("moving focal loss", gt_labels_one_hot, weight, gt_class_s, logits_class_t)
        loss_class = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_s,
                                                            logits=logits_class_t,
                                                            weights=weight)
        pass
    elif kwargs.get('class_loss') == "focal":
        gt_labels_one_hot = tf.one_hot(gt_class_s, 7, axis=1)
        weight = 1.0 - tf.reduce_sum(tf.multiply(gt_labels_one_hot,
                                                 pred_class), axis=1)
        weight = weight*weight
        print("focal loss", gt_labels_one_hot, weight, gt_class_s, logits_class_t)
        loss_class = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_s,
                                                            logits=logits_class_t,
                                                            weights=weight)
    elif kwargs.get('class_loss') == "ce":
        print("ce loss", gt_class_s, logits_class_t)
        loss_class = tf.losses.sparse_softmax_cross_entropy(gt_class_s,
                                                            logits_class_t)
    else:
        raise RuntimeError("invalid class loss! {}".format(kwargs.get('class_loss')))
    # loss_class = tf.Print(loss_class, [gt_class_s, logits_class_t,
    #                                    tf.math.reduce_min(gt_class_s),
    #                                    tf.math.reduce_max(gt_class_s)],
    #                       message="loss_class",
    #                       first_n=50, summarize=512)


    code = tf.transpose(tf.sigmoid(logits_code), [0, 2, 3, 1])
    sample_cnt = 1024 * kwargs['batch_size']
    gt_fgbgTmp = tf.squeeze(gt_fgbg, 1)
    # gt_fgbgTmp = tf.identity(gt_fgbg)
    gt_fg_loc = tf.where(gt_fgbgTmp)
    samples_loc = tf.random.uniform(
        (tf.math.minimum(sample_cnt, tf.shape(gt_fg_loc)[0]),),
        minval=0,
        maxval=tf.shape(gt_fg_loc)[0],
        dtype=tf.int32)
    samples_loc = tf.gather(gt_fg_loc, samples_loc)
    code_samples = tf.gather_nd(code, samples_loc)
    print(tf.transpose(gt_affs, [0, 2, 3, 1]))
    print(samples_loc)
    gt_affs_samples = tf.gather_nd(tf.transpose(gt_affs, [0, 2, 3, 1]),
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
    print("class loss factor 0.1")
    loss = losspatch + loss_fgbg + 0.1 * loss_class
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
    boundaries = [30000, 100000]
    values = [kwargs['lr'], kwargs['lr']/10, kwargs['lr']/100]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries,
                                                values)
    print(learning_rate)
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss,
                             global_step=global_step)

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
