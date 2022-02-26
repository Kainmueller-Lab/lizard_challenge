"""https://raw.githubusercontent.com/TissueImageAnalytics/CoNIC/main/compute_stats.py
compute_stats.py. Calculates the statistical measurements for the CoNIC Challenge.

This code supports binary panoptic quality for binary segmentation, multiclass panoptic quality for
simultaneous segmentation and classification and multiclass coefficient of determination (R2) for
multiclass regression. Binary panoptic quality is calculated per image and the results are averaged.
For multiclass panoptic quality, stats are calculated over the entire dataset for each class before taking
the average over the classes.

Usage:
    compute_stats.py [--mode=<str>] [--pred=<path>] [--true=<path>]
    compute_stats.py (-h | --help)
    compute_stats.py --version

Options:
    -h --help                   Show this string.
    --version                   Show version.
    --mode=<str>                Choose either `regression` or `seg_class`.
    --pred=<path>               Path to the results directory.
    --true=<path>               Path to the ground truth directory.

"""


import os
import pathlib
import shutil

import h5py
import zarr
from docopt import docopt
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label

def main():
    args = docopt(__doc__, version="CoNIC-stats-v1.0")

    mode = args["--mode"]
    pred_path = args["--pred"]
    true_path = args["--true"]

    work(mode, pred_path, true_path)


def work(mode, pred_path, true_path):
    seg_metrics_names = ["pq", "multi_pq+"]
    reg_metrics_names = ["r2"]

    # do initial checks
    if mode not in ["regression", "seg_class"]:
        raise ValueError("`mode` must be either `regression` or `seg_class`")

    semantic = "lorenz"
    # semantic = None

    hacks = True

    all_metrics = {}
    if mode == "seg_class":
        # check to make sure input is a single numpy array
        pred_format = pred_path.split(".")[-1]
        true_format = true_path.split(".")[-1]
        # if pred_format != "npy" or true_format != "npy":
        #     raise ValueError("pred and true must be in npy format.")

        # initialise empty placeholder lists
        pq_list = []
        mpq_info_list = []
        # load the prediction and ground truth arrays
        # pred_array = np.load(pred_path)
        # true_array = np.load(true_path)

        # nr_patches = pred_array.shape[0]
        preds = os.listdir(pred_path)
        nr_patches = len(preds)
        #nr_patches = 10

        if semantic == "lorenz":
            # preds_class = np.load("/home/peter/data/datasets/data_conic/pred_class_val_peter.npy")
            preds_class = np.load("/home/peter/data/datasets/data_conic/pred_class_val_peter2.npy")
            fls = os.listdir(true_path)
            flsT = []
            for fl in fls:
                flsT.append(os.path.splitext(fl)[0] + ".hdf")
            preds_classT = []
            # preds_classT2 = []
            for fl in preds:
                preds_classT.append(preds_class[flsT.index(fl)])
                # preds_classT2.append(flsT[flsT.index(fl)])

        mpq_info_per_image = {}
        for patch_idx in tqdm(range(nr_patches)):
        # for patch_idx in range(nr_patches):
            with h5py.File(os.path.join(pred_path, preds[patch_idx]), 'r') as f:
                # pred_inst = np.array(f['vote_isntances_dil_1'])
                pred_inst = np.array(f['vote_instances'])
            pred_inst = np.squeeze(pred_inst, axis=0)


            if hacks:
                pred_inst = remove_big_objects(pred_inst, size=5000)
                pred_inst = remove_holes(pred_inst, max_hole_size=50)
                pred_inst = instance_wise_connected_components(pred_inst)
                pred_inst = remove_small_objects(pred_inst, 30)
            # print(pred_path)

            label_path = pred_path.split("/")
            # print(label_path)
            label_path = os.path.join("/".join(label_path[:(label_path.index("instanced"))]),
                                      "processed",
                                      label_path[label_path.index("instanced")+1])
            # print(label_path)
            label_path = os.path.join(label_path, preds[patch_idx])
            # print(label_path)


            if semantic == "lorenz":
                pred_class = preds_classT[patch_idx]

                # pred_class5 = pred_class[5]
                # pred_class5[pred_class5 > 0.1] = 1.0
                # pred_class[5] = pred_class5
                # pred_class1 = pred_class[1]
                # pred_class1[pred_class1 > 0.1] = 1.0
                # pred_class[1] = pred_class1
                # print(list(np.max(pred_class,axis=(1,2))))

                pred_class = np.argmax(pred_class, axis=0)

            else:
                with h5py.File(label_path, 'r') as f:

                    pred_class = np.array(f['volumes/pred_labels'])
                    # print(pred_class.shape)
                # pred_classT = np.zeros_like(pred_inst)
                # continue


            true_inst = np.array(
                zarr.open(os.path.join(
                    true_path,
                    os.path.splitext(preds[patch_idx])[0] + ".zarr"), 'r')['volumes/gt_instances'])
            true_class = np.array(
                zarr.open(os.path.join(
                    true_path,
                    os.path.splitext(preds[patch_idx])[0] + ".zarr"), 'r')['volumes/gt_class'])

            if pred_inst.shape != pred_class.shape:
                print(pred_inst.shape, pred_class.shape, preds[patch_idx])
            #     pred_inst = pred_inst[:256, :256]


            pred_classT = np.zeros_like(pred_inst)
            for i in np.unique(pred_inst):
                if i == 0:
                    continue
                ids, cnts = np.unique(pred_class[pred_inst==i], return_counts=True)
                if ids[0] == 0:
                    ids = ids[1:]
                    cnts = cnts[1:]
                if len(ids) == 0:
                    continue
                k = ids[np.argmax(cnts)]
                # print(ids, cnts, k)
                pred_classT[pred_inst==i] = k

                # filter small stuff
                # if np.sum(pred_inst==i) > 30:
                #     pred_classT[pred_inst==i] = k
                # else:
                #     # print("removing instance")
                #     pred_inst[pred_inst==i] = 0

            pred_class = pred_classT

            true_inst = np.squeeze(true_inst, axis=0)
            true_class = np.squeeze(true_class, axis=0)

            # print(pred_inst.shape, pred_class.shape, true_inst.shape, true_class.shape)
            true = np.stack([true_inst, true_class], axis=-1)
            pred = np.stack([pred_inst, pred_class], axis=-1)
            # get a single patch
            # pred = pred_array[patch_idx]
            # true = true_array[patch_idx]

            # instance segmentation map
            # pred_inst = pred[..., 0]
            # true_inst = true[..., 0]

            # classification map
            # pred_class = pred[..., 1]
            # true_class = true[..., 1]

            # ===============================================================

            for idx, metric in enumerate(seg_metrics_names):
                if metric == "pq":
                    # get binary panoptic quality
                    pq = get_pq(true_inst, pred_inst)
                    pq = pq[0][2]
                    pq_list.append(pq)
                elif metric == "multi_pq+":
                    # get the multiclass pq stats info from single image
                    mpq_info_single = get_multi_pq_info(true, pred)
                    mpq_info = []
                    # aggregate the stat info per class
                    for idx2, single_class_pq in enumerate(mpq_info_single):
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])

                        dq = tp / (
                            (tp + 0.5 * fp + 0.5 * fn) + 1.0e-6
                        )
                        # get the SQ, when not paired, it has 0 IoU so does not impact
                        sq = sum_iou / (tp + 1.0e-6)
                        mpq_info_per_image.setdefault(idx2, []).append(dq * sq)

                        # print(patch_idx, idx, idx2, [len(l) for l in mpq_info_per_image.values()], dq * sq)
                        # print(mpq_info_per_image)
                    mpq_info_list.append(mpq_info)
                else:
                    raise ValueError("%s is not supported!" % metric)

        mpq_info_per_imageT = []
        for k in sorted(mpq_info_per_image.keys()):
            mpq_info_per_imageT.append(mpq_info_per_image[k])
        mpq_info_per_image = np.array(mpq_info_per_imageT)
        print(mpq_info_per_image.shape)
        print(np.mean(mpq_info_per_image, axis=0))
        print(sorted(zip(np.mean(mpq_info_per_image, axis=0), preds)))


        pq_metrics = np.array(pq_list)
        pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images
        if "multi_pq+" in seg_metrics_names:
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # sum over all the images
            total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

        for idx, metric in enumerate(seg_metrics_names):
            if metric == "multi_pq+":
                mpq_list = []
                # for each class, get the multiclass PQ
                for cat_idx in range(total_mpq_info_metrics.shape[0]):
                    total_tp = total_mpq_info_metrics[cat_idx][0]
                    total_fp = total_mpq_info_metrics[cat_idx][1]
                    total_fn = total_mpq_info_metrics[cat_idx][2]
                    total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                    # get the F1-score i.e DQ
                    dq = total_tp / (
                        (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                    )
                    # get the SQ, when not paired, it has 0 IoU so does not impact
                    sq = total_sum_iou / (total_tp + 1.0e-6)
                    mpq_list.append(dq * sq)
                mpq_metrics = np.array(mpq_list)
                print(mpq_metrics)
                all_metrics[metric] = [np.mean(mpq_metrics)]
            else:
                all_metrics[metric] = [pq_metrics_avg]

    else:
        # first check to make sure ground truth and prediction is in csv format
        if not os.path.isfile(true_path) or not os.path.isfile(pred_path):
            raise ValueError("pred and true must be in csv format.")

        pred_format = pred_path.split(".")[-1]
        true_format = true_path.split(".")[-1]
        if pred_format != "csv" or true_format != "csv":
            raise ValueError("pred and true must be in csv format.")

        pred_csv = pd.read_csv(pred_path)
        true_csv = pd.read_csv(true_path)

        for idx, metric in enumerate(reg_metrics_names):
            if metric == "r2":
                # calculate multiclass coefficient of determination
                r2 = get_multi_r2(true_csv, pred_csv)
                all_metrics["multi_r2"] = [r2]
            else:
                raise ValueError("%s is not supported!" % metric)

    df = pd.DataFrame(all_metrics)
    df = df.to_string(index=False)
    print(df)
    return all_metrics


# https://raw.githubusercontent.com/TissueImageAnalytics/CoNIC/main/metrics/stats_utils.py
def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.

    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time,
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.

    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        nr_classes (int): Number of classes considered in the dataset.
        match_iou (float): IoU threshold for determining whether there is a detection.

    Returns:
        statistical info per class needed to compute PQ.

    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)

    return pq


def get_pq(true, pred, match_iou=0.5, remap=True):
    """Get the panoptic quality result.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag
    has no effect on the result.

    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5,
            Munkres assignment (solving minimum weight matching in bipartite graphs)
            is caculated to find the maximal amount of unique pairing. If
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.
        remap (bool): whether to ensure contiguous ordering of instances.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

        paired_iou.sum(): sum of IoU within true positive predictions

    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
    )


def get_multi_r2(true, pred):
    """Get the correlation of determination for each class and then
    average the results.

    Args:
        true (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
        pred (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.

    Returns:
        multi class coefficient of determination

    """
    # first check to make sure that the appropriate column headers are there
    class_names = [
        "epithelial",
        "lymphocyte",
        "plasma",
        "neutrophil",
        "eosinophil",
        "connective",
    ]
    for col in true.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    for col in pred.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    # for each class, calculate r2 and then take the average
    r2_list = []
    for class_ in class_names:
        true_oneclass = true[class_].tolist()
        pred_oneclass = pred[class_].tolist()
        r2_list.append(r2_score(true_oneclass, pred_oneclass))

    return np.mean(np.array(r2_list))


# https://github.com/TissueImageAnalytics/CoNIC/blob/main/misc/utils.py
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).
    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.
    Args:
        img: input binary image.
    Returns:
        bounding box coordinates
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def instance_wise_connected_components(pred_inst, connectivity=2):
    out = np.zeros_like(pred_inst)
    i = np.max(pred_inst)
    for j in np.unique(pred_inst):
        if j ==0:
            continue
        relabeled = label(pred_inst==j, background=0, connectivity=connectivity)
        first = True
        for new_lab in np.unique(relabeled):
            if new_lab == 0:
                continue
            if first:
                out[relabeled==new_lab] = j
                first = False
            if np.sum(relabeled==new_lab)>40:
                out[relabeled==new_lab] = i
                i += 1
            else:
                out[relabeled==new_lab] = j
    return out

def remove_big_objects(pred_inst, size):
    for i in np.unique(pred_inst):
        if i == 0:
            continue
        if np.sum(pred_inst==i)>size:
            print('Remove instance '+str(i)+' of size '+str(np.sum(pred_inst==i)))
            pred_inst[pred_inst==i] = 0
    return pred_inst

def remove_holes(pred_inst, max_hole_size):
    out = np.zeros_like(pred_inst)
    for i in np.unique(pred_inst):
        if i == 0:
            continue
        out += remove_small_holes(pred_inst==i, max_hole_size)*i
    return out


if __name__ == "__main__":
    main()
