from torch.utils.data import Dataset
import numpy as np
import mahotas
import scipy
import logging
import torch

from skimage.measure import label
from skimage.morphology import remove_small_holes

def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-8, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x   = x.astype(dtype, copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x, 0, 1)
    return x

class SliceDataset(Dataset):
    def __init__(self, raw, labels):
        self.raw = raw
        self.labels = labels
    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, idx):
        raw_tmp = normalize_percentile(self.raw[idx].astype(np.float32))
        if self.labels is not None:
            return raw_tmp, self.labels[idx].astype(np.float32)
        else:
            return raw_tmp, False

logger = logging.getLogger(__name__)

def watershed(surface, markers, fg):
    logger.info("labelling")
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    logger.debug("watershed output: %s %s %f %f",
                 ws.shape, ws.dtype, ws.max(), ws.min())

    # overlay fg and write
    wsFG = ws * fg
    logger.debug("watershed (foreground only): %s %s %f %f",
                 wsFG.shape, wsFG.dtype, wsFG.max(),
                 wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    return wsFGUI

def make_instance_segmentation(prediction, fg_thresh=0.9, seed_thresh=0.9):
    # prediction[0] = bg
    # prediction[1] = inside
    # prediction[2] = boundary
    fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
    ws_surface = 1.0 - prediction[1, ...]
    seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
    markers, cnt = scipy.ndimage.label(seeds)
    labelling = watershed(ws_surface, markers, fg)
    return labelling, ws_surface

def make_pseudolabel(raw, model, n_views, slow_aug):
    B,C,H,w = raw.shape
    mask_list = []
    out_list = []
    for b in range(B): 
        tmp_out_list = []
        tmp_mask_list = []
        for _ in range(n_views):
            # gen views
            slow_aug.interpolation='bilinear'
            view = slow_aug.forward_transform(raw[b].unsqueeze(0))
            with torch.no_grad():
                out = model(view)
            mask = torch.ones_like(out[:,-1:,:,:])
            slow_aug.interpolation='nearest'
            out_inv, aug_mask_inv = slow_aug.inverse_transform(out, mask)
            tmp_out_list.append(out_inv*aug_mask_inv)
            tmp_mask_list.append(aug_mask_inv)
        out_slow = torch.stack(tmp_out_list).sum(0) # 1 x 3 x H x W
        mask_slow = torch.stack(tmp_mask_list).sum(0) > 0 # 1 x 1 x H x W
        n_out = torch.stack(tmp_mask_list).sum(0)
        out_slow = mask_slow*out_slow/(n_out+1e-6)
        out_list.append(out_slow)
        mask_list.append(mask_slow)
    out = torch.cat(out_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    return out, mask

def center_crop(t, croph, cropw):
    h,w = t.shape
    startw = w//2-(cropw//2)
    starth = h//2-(croph//2)
    return t[starth:starth+croph,startw:startw+cropw]


def make_ct(pred_class, instance_map):
    device = pred_class.device
    pred_ct = torch.zeros_like(instance_map)
    pred_class_tmp = pred_class.softmax(1).squeeze(0)
    for instance in instance_map.unique():
        if instance==0:
            continue
        ct = pred_class_tmp[:,instance_map==instance].sum(1)
        ct = ct.argmax()
        pred_ct[instance_map==instance] = ct
    # actually redo this for the center crop of the image
    ct_list = np.zeros(7)
    instance_map_tmp = center_crop(instance_map, 224,224)
    for instance in instance_map_tmp.unique():
        if instance==0:
            continue
        ct_tmp = pred_ct[instance_map==instance]
        ct_list[ct_tmp.detach().cpu().numpy()] += 1
    pred_reg = {
        "neutrophil"            : ct_list[1],
        "epithelial-cell"       : ct_list[2],
        "lymphocyte"            : ct_list[3],
        "plasma-cell"           : ct_list[4],
        "eosinophil"            : ct_list[5],
        "connective-tissue-cell": ct_list[6],
    }
    return pred_ct, pred_reg



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