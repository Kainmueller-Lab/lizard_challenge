from torch.utils.data import Dataset
import numpy as np

import scipy
import scipy.special
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
            return torch.from_numpy(raw_tmp), False

logger = logging.getLogger(__name__)

# def watershed(surface, markers, fg):
#     logger.info("labelling")
#     # compute watershed
#     ws = mahotas.cwatershed(surface, markers)

#     # write watershed directly
#     logger.debug("watershed output: %s %s %f %f",
#                  ws.shape, ws.dtype, ws.max(), ws.min())

#     # overlay fg and write
#     wsFG = ws * fg
#     logger.debug("watershed (foreground only): %s %s %f %f",
#                  wsFG.shape, wsFG.dtype, wsFG.max(),
#                  wsFG.min())
#     wsFGUI = wsFG.astype(np.uint16)

#     return wsFGUI

def make_instance_segmentation(prediction, fg_thresh=0.9, seed_thresh=0.9):
    # prediction[0] = bg
    # prediction[1] = inside
    # prediction[2] = boundary
    fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
    ws_surface = 1.0 - prediction[1, ...]
    seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
    markers, cnt = scipy.ndimage.label(seeds)
    labelling = None #watershed(ws_surface, markers, fg)
    return labelling, ws_surface


rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = np.linalg.inv(rgb_from_hed)


hed_t = torch.tensor(hed_from_rgb, dtype=torch.float)
rgb_t = torch.tensor(rgb_from_hed, dtype=torch.float)

def torch_rgb2hed(img, hed_t, e):
    img = img.T
    img = torch.clamp(img, min = e)
    img = torch.log(img)/torch.log(e)
    img = torch.matmul(img, hed_t)
    return img.T

def torch_hed2rgb(img, rgb_t, e):
    e = -torch.log(e)
    img = img.T
    img = torch.matmul(-(img*e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0,1)
    return img.T


class Hed2Rgb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = torch.tensor(1e-6)
        self.rgb_t = rgb_t
    
    def forward(self, img):
        r = img.device
        if img.dim()==3:
            return torch_hed2rgb(img, self.rgb_t.to(r), self.e.to(r))
        else:
            out = []
            for i in img:
                out.append(torch_hed2rgb(i, self.rgb_t.to(r), self.e.to(r)))
            return torch.stack(out)

class Rgb2Hed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = torch.tensor(1e-6)
        self.hed_t = hed_t
    
    def forward(self, img):
        r = img.device
        if img.dim()==3:
            return torch_rgb2hed(img, self.hed_t.to(r), self.e.to(r))
        else:
            out = []
            for i in img:
                out.append(torch_rgb2hed(i, self.hed_t.to(r), self.e.to(r)))
            return torch.stack(out)


def linear_contrast(img, alpha, e):
    return e + alpha*(img-e)

class LinearContrast(torch.nn.Module):
    '''
    Based on Imgaug linear contrast

    alpha: tuple of floats
        e.g. (0.95,1.05)

    per_channel: bool
        whether to apply different augmentation per channel

    e: float
        should be the mean value of the transformed image (for HED transformed images, this is not really 0)
    '''
    def __init__(self, alpha, per_channel=True, e=.0):
        super().__init__()
        self.alpha = alpha 
        self.s = (3,1,1) if per_channel else (1)        
        self.e = torch.tensor(e)
       

    def forward(self, img):
        if img.dim()==3:
            a = torch.empty(self.s).uniform_(self.alpha[0], self.alpha[1]).to(img.device)
            return linear_contrast(img, a, self.e.to(img.device))
        else:
            out = []
            for i in img:
                a = torch.empty(self.s).uniform_(self.alpha[0], self.alpha[1]).to(img.device)
                out.append(linear_contrast(i, a, self.e.to(img.device)))
            return torch.stack(out)

def color_augmentations():
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    HED_contrast = torch.nn.Sequential(
                Rgb2Hed(),
                LinearContrast(alpha=(0.9,1.1)),
                Hed2Rgb())
    data_transforms = torch.nn.Sequential(
            HED_contrast
    )
    return data_transforms

color_aug_fn = color_augmentations()

def make_pseudolabel(raw, model, n_views, slow_aug, reduce='mean'):
    B,C,H,w = raw.shape
    ct_list = []
    inst_list = []
    cpv_list = []
    for b in range(B): 
        tmp_ct_list = []
        tmp_inst_list = []
        tmp_cpv_list = []
        for _ in range(n_views):
            # gen views
            slow_aug.interpolation='bilinear'
            view = slow_aug.forward_transform(raw[b].unsqueeze(0))
            view = color_aug_fn(view)
            with torch.no_grad():
                out = model(view)
            mask = torch.ones_like(out[:,-1:,:,:])
            slow_aug.interpolation='nearest'
            out_inv, aug_mask_inv = slow_aug.inverse_transform(out, mask)

            cpv,_ = out_inv[:,:2].abs().max(1)
            if reduce =='sum':
                pred_3c = out_inv[:,2:5]
                pred_ct = out_inv[:,5:]
            else:
                pred_3c = out_inv[:,2:5].softmax(1)
                pred_ct = out_inv[:,5:].softmax(1)
            tmp_inst_list.append(pred_3c*aug_mask_inv)
            tmp_ct_list.append(pred_ct*aug_mask_inv)
            tmp_cpv_list.append(cpv)
        
        if reduce =='max':
            pred_inst,_ = torch.stack(tmp_inst_list).max(0) # 1 x 3 x H x W
            pred_ct,_ = torch.stack(tmp_ct_list).max(0) # 1 x 3 x H x W
            cpv = torch.stack(tmp_cpv_list).max(0) # 1 x 3 x H x W
        elif reduce =='mean':
            pred_inst = torch.stack(tmp_inst_list).mean(0) # 1 x 3 x H x W
            pred_ct = torch.stack(tmp_ct_list).mean(0) # 1 x 3 x H x W
            cpv = torch.stack(tmp_cpv_list).mean(0) # 1 x 3 x H x W
        elif reduce =='sum':
            pred_inst = torch.stack(tmp_inst_list).sum(0).softmax(1) # 1 x 3 x H x W
            pred_ct = torch.stack(tmp_ct_list).sum(0).softmax(1) # 1 x 3 x H x W
            cpv = torch.stack(tmp_cpv_list).sum(0) # 1 x 3 x H x W
            
        ct_list.append(pred_ct)
        inst_list.append(pred_inst)
        cpv_list.append(cpv)
    ct = torch.cat(ct_list, dim=0)
    inst = torch.cat(inst_list, dim=0)
    cpv = torch.cat(cpv_list, dim=0)
    return ct, inst, cpv


def center_crop(t, croph, cropw):
    h,w = t.shape
    startw = w//2-(cropw//2)
    starth = h//2-(croph//2)
    return t[starth:starth+croph,startw:startw+cropw]


def make_ct(pred_class, instance_map):
    # device = pred_class.device
    pred_ct = np.zeros_like(instance_map)
    # print(pred_class.shape)
    pred_class_tmp = np.squeeze(scipy.special.softmax(pred_class, axis=1), axis=0)
    pred_class_tmp2 = np.argmax(pred_class_tmp, axis=0)
    # print(pred_class_tmp.shape)
    # print(instance_map.unique().detach().cpu().numpy())
    softmax = False
    for instance in np.unique(instance_map):
        if instance==0:
            continue
        if softmax:
            ct_t = np.sum(pred_class_tmp[:,instance_map==instance], axis=1)
            ct = np.argmax(ct_t)
            # print(ct_t)
            #if ct == 0:
            #    ct = ct_t[1:].argmax()
        else:
            ids, cnts = np.unique(pred_class_tmp2[instance_map==instance],
                                  return_counts=True)
            if ids[0] == 0:
                ids = ids[1:]
                cnts = cnts[1:]
            if len(ids) == 0:
                ct_t = pred_class_tmp[:,instance_map==instance].sum(1)
                ct = np.argmax(ct_t)
                if ct == 0:
                    ct = np.argmax(ct_t[1:])
                # print(ct_t)
            ct = ids[np.argmax(cnts)]
        pred_ct[instance_map==instance] = ct
    # actually redo this for the center crop of the image
    ct_list = np.zeros(7)
    instance_map_tmp = center_crop(instance_map, 224,224)
    for instance in np.unique(instance_map_tmp):
        if instance==0:
            continue
        ct_tmp = pred_ct[instance_map==instance]
        ct_list[ct_tmp] += 1
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
