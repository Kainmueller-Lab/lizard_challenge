import itk
import numpy as np

import pandas as pd
from tqdm.auto import tqdm
from src.metrics import *

def center_crop(t, croph, cropw):
    h,w = t.shape
    startw = w//2-(cropw//2)
    starth = h//2-(croph//2)
    return t[starth:starth+croph,startw:startw+cropw]

def calc_MPQ():
    mode = 'seg_class'
    pred_array = np.array(itk.imread("/home/buzzwoll/lizard_challenge/docker_submission/local_test/output/pred_seg.mha"))
    true_array = np.load("/home/buzzwoll/lizard_challenge/docker_submission/local_test/Y_val200.npy")

    seg_metrics_names = ["pq", "multi_pq+"]
    reg_metrics_names = ["r2"]

    all_metrics = {}
    if mode == "seg_class":
        # check to make sure input is a single numpy array
    #     pred_format = pred_path.split(".")[-1]
    #     true_format = true_path.split(".")[-1]
    #     if pred_format != "npy" or true_format != "npy":
    #         raise ValueError("pred and true must be in npy format.")

        # initialise empty placeholder lists
        pq_list = []
        mpq_info_list = []
        # load the prediction and ground truth arrays
        #pred_array = np.load(pred_path)
        #true_array = np.load(true_path)

        nr_patches = pred_array.shape[0]

        for patch_idx in tqdm(range(nr_patches)):
            # get a single patch
            pred = pred_array[patch_idx]
            true = true_array[patch_idx]

            # instance segmentation map
            pred_inst = pred[..., 0]
            true_inst = true[..., 0]
            # classification map
            pred_class = pred[..., 1]
            true_class = true[..., 1]

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
                    for single_class_pq in mpq_info_single:
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)
                else:
                    raise ValueError("%s is not supported!" % metric)

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
                all_metrics[metric] = [np.mean(mpq_metrics)]
            else:
                all_metrics[metric] = [pq_metrics_avg]

    df = pd.DataFrame(all_metrics)
    print(df)
    print(mpq_list)
    return df, mpq_list


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
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "neutrophil",
        "eosinophil",
        "connective-tissue-cell",
    ]
    for col in true.keys():
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    for col in pred.keys():
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    # for each class, calculate r2 and then take the average
    r2_list = []
    for class_ in class_names:
        true_oneclass = true[class_]
        pred_oneclass = pred[class_]
        r2_list.append(r2_score(true_oneclass, pred_oneclass))
    print(r2_list)
    return np.mean(np.array(r2_list)), r2_list



def make_reg(pred_ct, instance_map):
    ct_list = [0,0,0,0,0,0,0]
    instance_map_tmp = center_crop(instance_map, 224,224)
    for instance in np.unique(instance_map_tmp):
        if instance==0:
            continue
        ct_tmp = pred_ct[instance_map==instance]
        idx = int(ct_tmp.max())
        ct_list[idx] += 1
    pred_reg = {
        "neutrophil"            : ct_list[1],
        "epithelial-cell"       : ct_list[2],
        "lymphocyte"            : ct_list[3],
        "plasma-cell"           : ct_list[4],
        "eosinophil"            : ct_list[5],
        "connective-tissue-cell": ct_list[6],
    }
    return pred_reg



def calc_R2(gt_list):
    gt_regression = {
    "neutrophil"            : [],
    "epithelial-cell"       : [],
    "lymphocyte"            : [],
    "plasma-cell"           : [],
    "eosinophil"            : [],
    "connective-tissue-cell": [],
    }

    for gt in tqdm(gt_list):
        gt_inst, gt_ct = gt[...,0], gt[...,1]
        ct_list = np.zeros(7)
        instance_map_tmp = center_crop(gt_inst, 224,224)
        for instance in np.unique(instance_map_tmp):
            if instance==0:
                continue
            ct_tmp = gt_ct[gt_inst==instance][0]
            ct_list[int(ct_tmp)] += 1
        gt_reg = {
            "neutrophil"            : ct_list[1],
            "epithelial-cell"       : ct_list[2],
            "lymphocyte"            : ct_list[3],
            "plasma-cell"           : ct_list[4],
            "eosinophil"            : ct_list[5],
            "connective-tissue-cell": ct_list[6],
        }
        for key in gt_regression.keys():
            gt_regression[key].append(gt_reg[key])

    for key in gt_regression.keys():
        gt_regression[key] = np.array(gt_regression[key])

    r2, r2_list = get_multi_r2(gt_regression, pred_regression)
    print('R2: ', r2)
    return r2, r2_list

pred_array = np.array(itk.imread("/home/buzzwoll/lizard_challenge/docker_submission/local_test/output/pred_seg.mha"))
true_array = np.load("/home/buzzwoll/lizard_challenge/docker_submission/local_test/Y_val200.npy")
pred_regression = {
    "neutrophil"            : [],
    "epithelial-cell"       : [],
    "lymphocyte"            : [],
    "plasma-cell"           : [],
    "eosinophil"            : [],
    "connective-tissue-cell": [],
}
for pred_inst, pred_class in tqdm(zip(pred_array[...,0], pred_array[...,1])):
    pred_reg = make_reg(pred_class,pred_inst)
    for key in pred_regression.keys():
        pred_regression[key].append(pred_reg[key])

for key in pred_regression.keys():
    pred_regression[key] = np.array(pred_regression[key])


calc_R2(true_array)
# calc_MPQ()
print("test")

