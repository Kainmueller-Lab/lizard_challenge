from __future__ import print_function
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import tensorflow as tf
import zarr

import gunpowder as gp

logger = logging.getLogger(__name__)


def train_until(**kwargs):
    print("cuda visibile devices", os.environ["CUDA_VISIBLE_DEVICES"])
    if tf.train.latest_checkpoint(kwargs['output_folder']):
        trained_until = int(
            tf.train.latest_checkpoint(kwargs['output_folder']).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= kwargs['max_iteration']:
        return

    anchor = gp.ArrayKey('ANCHOR')
    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    gt_labels = gp.ArrayKey('GT_LABELS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    gt_fgbg = gp.ArrayKey('GT_FGBG')
    gt_class = gp.ArrayKey('GT_CLASS')

    #loss_weights_fgbg = gp.ArrayKey('LOSS_WEIGHTS_FGBG')

    pred_code = gp.ArrayKey('PRED_CODE')
    pred_fgbg = gp.ArrayKey('PRED_FGBG')
    pred_class = gp.ArrayKey('PRED_CLASS')
    pred_code_gradients = gp.ArrayKey('PRED_CODE_GRADIENTS')

    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name'] + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name']  + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape'])*voxel_size

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_shape_world)
    request.add(raw_cropped, output_shape_world)
    request.add(gt_labels, output_shape_world)
    request.add(anchor, output_shape_world)
    request.add(gt_affs, output_shape_world)
    request.add(gt_fgbg, output_shape_world)
    request.add(gt_class, output_shape_world)
    #request.add(loss_weights_fgbg, output_shape_world)

    # when we make a snapshot for inspection (see below), we also want to
    # request the predicted affinities and gradients of the loss wrt the
    # affinities
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw_cropped, output_shape_world)
    snapshot_request.add(pred_code, output_shape_world)
    snapshot_request.add(pred_fgbg, output_shape_world)
    snapshot_request.add(pred_class, output_shape_world)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("train node for %s not implemented yet",
                                  kwargs['input_format'])

    raw_key = kwargs.get('raw_key', 'volumes/raw')
    gt_key = kwargs.get('gt_key', 'volumes/gt_instances')

    fls = []
    fls_pcs = []
    pcs_sum = 0
    for f in sorted(kwargs['data_files']):
        px = zarr.open(f, 'r')[raw_key].shape[1:]
        pcs = px[0] * px[1]
        pcs_sum += pcs
        fls_pcs.append(pcs)
        fls.append(os.path.splitext(f)[0])

    # fls_pcs = [0.003892465515043975, 0.009438303510358487, 0.003510693454569542, 0.019497578939079656, 0.004231621259996586, 0.013826554704652176, 0.005375202000259019, 0.01171356395936793, 0.006112453859192768, 0.005025643619736879, 0.0038320291209403526, 0.016079451635256578, 0.028704643969913714, 0.03324188224582915, 0.05920522180379596, 0.12246975304061586, 0.13028696074997717, 0.08751672123792949, 0.07014707280327666, 0.03815937040696391, 0.07263135028655077, 0.061486589237697294, 0.04338627061226907, 0.040700369325122075, 0.12320945144528442, 0.017697771217607697, 0.06798153482063442, 0.0453503778313735, 0.02534973788923839, 0.049519972368958956, 0.055656162105594056, 0.04936181444507461, 0.09324895980938228, 0.04376675743231963, 0.03855670332109939, 0.07371584947516138, 0.0324481611512907, 0.042778092277937045, 0.12744908638758956, 0.03852773723876765, 0.045264049510865455, 0.027036960307660708, 0.01868639582944706, 0.05825665904235185, 0.15235587754194607, 0.14857911400023577, 0.0546058134022513, 0.07165041155150786, 0.025928322403017484, 0.043947306432228064, 0.06862839307416345, 0.018218817422057977, 0.050126648677053985, 0.0532066233805908, 0.047875298734984494, 0.035345073789829345, 0.18100363346081477, 0.05871791203545766, 0.040767501000314864, 0.04433967188464218, 0.06167122226760792, 0.027652029719423852, 0.06449539640843577, 0.0761568855259939, 0.06148519399226071, 0.047759038218511816, 0.095596065166102, 0.03416480562627135, 0.03622763194875641, 0.02716475684222877, 0.062193025991233254, 0.08525205580199725, 0.07678292785215025, 0.0320455221760161, 0.08420930509812431, 0.019191974825205606, 0.22836568600892052, 0.013847310339362798, 0.0296638417517546, 0.03902498210107279, 0.043922812095713894, 0.02123341799839016, 0.04190419725115655, 0.1475757699485203, 0.017529817556148023, 0.02447577116601228, 0.015093726089121212, 0.055478787667445725, 0.06344829784921319, 0.09420476349038502, 0.028485916508431693, 0.03726685833451511, 0.03197343022426974, 0.22853322963829759, 0.11746272282798355, 0.05170571225040037, 0.027473764350679858, 0.040295593026498426, 0.11607508242433978, 0.025230058067902534, 0.016152548236998934, 0.01998022329338605, 0.09836047052656478, 0.040349962154947974, 0.017594877106676114, 0.036325259389100056, 0.057726117374122574, 0.029552583092694394, 0.048305759610533704, 0.035228565162973825, 0.02861604433806128, 0.08443343586683939, 0.1087428776466334, 0.02518069152078669, 0.06014002704127098, 0.09890932852480541, 0.028754528856082866, 0.007816503913949426, 0.012995926535634127, 0.01098123299712662, 0.012465826097108225, 0.006364038868523739, 0.01139109552272488, 0.006699392283821118, 0.008043200197485384, 0.010973820337739646, 0.01513611490211133, 0.008933909225700485, 0.009417862571048808, 0.009227985150042313, 0.00898562773212399, 0.007726585783847365, 0.010637267619404402, 0.009234554926232449, 0.011145194613005246, 0.013103620207465255, 0.01619017717795571, 0.017497902452255267, 0.01319598932191343, 0.007877079403448145, 0.012928211751419448, 0.007650313722321349, 0.009772490543468603, 0.019636222124991092, 0.00730410818271978, 0.010820657904777401, 0.009079914470745602, 0.011622451901353326, 0.007323686285442554, 0.009413221474946429, 0.012978884171091068, 0.007247170133146456, 0.005328763726989167, 0.007309519388773159, 0.008433253443184102, 0.02007586510972635, 0.005109644018050691, 0.00945644197539718, 0.009792353430166554, 0.014477871940496856, 0.007080476092143151, 0.010361967418568468, 0.008170984081371997, 0.015357814931300617, 0.008847214027774768, 0.0032113524637114453, 0.010394559862022007, 0.005722076638114426, 0.003644869446375592, 0.012672729946128914, 0.006313946103249314, 0.0110203511962275, 0.004089368210354728, 0.0055921384595316445, 0.003417735306228647, 0.0018359712941227555, 0.003986832375545609, 0.02274728181210085, 0.004255823754547696, 0.00949270467494442, 0.009023407468688306, 0.003362318709996358, 0.0040896051513452285, 0.00698933127488012, 0.009514812626999934, 0.009694016630811412, 0.007807438555916965, 0.009519789628745601, 0.0036780221519137103]
    print(len(fls_pcs), len(fls))



    ln = len(fls)
    print("first 5 files: ", fls[0:4], fls_pcs[:4], pcs_sum)

    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource

    neighborhood = []
    psH = np.array(kwargs['patchshape'])//2
    for i in range(-psH[1], psH[1]+1, kwargs['patchstride'][1]):
        for j in range(-psH[2], psH[2]+1, kwargs['patchstride'][2]):
            neighborhood.append([i,j])

    datasets = {
        raw: raw_key,
        gt_labels: gt_key,
        gt_fgbg: 'volumes/gt_fgbg',
        gt_class: 'volumes/gt_class',
        anchor: 'volumes/gt_fgbg',
    }
    array_specs = {
        raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
        gt_labels: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        gt_fgbg: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        gt_class: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        anchor: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
    }
    inputs = {
        net_names['raw']: raw,
        net_names['gt_affs']: gt_affs,
        net_names['gt_fgbg']: gt_fgbg,
        net_names['gt_class']: gt_class,
        net_names['anchor']: anchor,
        #net_names['loss_weights_fgbg']: loss_weights_fgbg,
    }

    outputs = {
        net_names['pred_code']: pred_code,
        net_names['pred_fgbg']: pred_fgbg,
        net_names['pred_class']: pred_class,
        net_names['raw_cropped']: raw_cropped,
    }
    snapshot = {
        raw: raw_key,
        raw_cropped: raw_key + '_cropped',
        gt_affs: '/volumes/gt_affs',
        gt_fgbg: '/volumes/gt_fgbg',
        gt_class: '/volumes/gt_class',
        pred_code: '/volumes/pred_code',
        pred_fgbg: '/volumes/pred_fgbg',
        pred_class: '/volumes/pred_class',
        # pred_code_gradients: '/volumes/pred_code_gradients',
    }

    augmentation = kwargs['augmentation']
    pipeline = (
        tuple(
            sourceNode(
                fls[t] + "." + kwargs['input_format'],
                datasets=datasets,
                array_specs=array_specs
            )
            + gp.Normalize(raw, factor=1.0/255.0)
            + gp.Pad(raw, None)
            + gp.Pad(gt_labels, None)
            + gp.Pad(gt_fgbg, None)
            + gp.Pad(gt_class, None)
            # + gp.Pad(gt_fgbg, (60, 60))
            #+ gp.Pad(anchor, (200, 200))

            # chose a random location for each requested batch
            + gp.RandomLocation()

            for t in range(ln)
        ) +

        # chose a random source (i.e., sample) from the above
        gp.RandomProvider(probabilities=fls_pcs) +

        # elastically deform the batch
        gp.ElasticAugment(
            augmentation['elastic']['control_point_spacing'],
            augmentation['elastic']['jitter_sigma'],
            [augmentation['elastic']['rotation_min']*np.pi/180.0,
             augmentation['elastic']['rotation_max']*np.pi/180.0],
            subsample=augmentation['elastic'].get('subsample', 1)) +

        gp.Reject(gt_fgbg, min_masked=0.002, reject_probability=1) +

        # apply transpose and mirror augmentations
        gp.SimpleAugment(mirror_only=augmentation['simple'].get("mirror"),
                         transpose_only=augmentation['simple'].get("transpose")) +

        # scale and shift the intensity of the raw array
        gp.IntensityAugment(
            raw,
            scale_min=augmentation['intensity']['scale'][0],
            scale_max=augmentation['intensity']['scale'][1],
            shift_min=augmentation['intensity']['shift'][0],
            shift_max=augmentation['intensity']['shift'][1],
            z_section_wise=False) +

        # gp.IntensityScaleShift(raw, 2, -1) +

        # convert labels into affinities between voxels
        gp.AddAffinities(
            neighborhood,
            gt_labels,
            gt_affs) +

        # create a weight array that balances positive and negative samples in
        # the affinity array
        #gp.BalanceLabels(
        #    gt_fgbg,
        #    loss_weights_fgbg) +


        gp.Stack(kwargs['batch_size']) +

        # RejectClass(gt_class, reject_probability=1.0) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=kwargs['cache_size'],
            num_workers=kwargs['num_workers']) +


        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            os.path.join(kwargs['output_folder'], kwargs['name']),
            optimizer=net_names['optimizer'],
            summary=net_names['summaries'],
            log_dir=kwargs['output_folder'],
            loss=net_names['loss'],
            inputs=inputs,
            outputs=outputs,
            gradients={
                net_names['pred_code']: pred_code_gradients,
            },
            save_every=kwargs['checkpoints']) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot(
            snapshot,
            output_dir=os.path.join(kwargs['output_folder'], 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=kwargs['snapshots'],
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=kwargs['profiling'])
    )

    #########
    # TRAIN #
    #########
    print("Starting training...")
    with gp.build(pipeline):
        # print(pipeline)
        for i in range(trained_until, kwargs['max_iteration']):
            # print("request", request)
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            print(
                "Batch: iteration=%d, time=%f" %
                (i, time_of_iteration))
            # exit()
    print("Training finished")


import logging
import random

from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class RejectClass(gp.BatchFilter):
    '''Reject batches based on the masked-in vs. masked-out ratio.

    Args:

        mask (:class:`ArrayKey`, optional):

            The mask to use, if any.

        min_masked (``float``, optional):

            The minimal required ratio of masked-in vs. masked-out voxels.
            Defaults to 0.5.

        ensure_nonempty (:class:`GraphKey`, optional)

            Ensures there is at least one point in the batch.

        reject_probability (``float``, optional):

            The probability by which a batch that is not valid (less than
            min_masked) is actually rejected. Defaults to 1., i.e. strict
            rejection.
    '''

    def __init__(
            self,
            gt_class,
            reject_probability=1.0):
        self.gt_class = gt_class
        self.reject_probability = reject_probability

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):
        random.seed(request.random_seed)

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        have_good_batch = False
        while not have_good_batch:

            timing.stop()
            batch = self.upstream_provider.request_batch(request)
            timing.start()

            gt_class = batch.arrays[self.gt_class].data

            classes = sorted(list(np.unique(gt_class)))
            have_good_batch = classes == list(range(7))

            if not have_good_batch and self.reject_probability < 1.:
                have_good_batch = random.random() > self.reject_probability
                if have_good_batch:
                    logger.debug("accepted bad batch (prob %s)", self.reject_probability)

            if not have_good_batch:
                num_rejected += 1
                logger.warning("reject batch, class missing %s", classes)

                if timing.elapsed() > report_next_timeout:

                    logger.warning(
                        "rejected %d batches, been waiting for a good one "
                        "since %ds", num_rejected, report_next_timeout)
                    report_next_timeout *= 2

        logger.warning("accept batch, classes %s", classes)
        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
