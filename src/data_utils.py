import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import zarr
from torch.utils.data import Dataset
from torch import nn
import random
from torchvision import transforms
from torchvision.transforms.transforms import RandomApply, GaussianBlur, ColorJitter
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import binary_erosion
from tqdm.auto import tqdm

class CombinedDataLoader():
    '''
    DataLoader which combines all the other DataLoaders in this project. 
    __next__ yields a batch of size batch_size containing samples from all DataLoaders,
    composed based on their sampling_probs

    dataloaders: list
        Containing torch.utils.data.DataLoader objects with batch_size=1
    batch_size: int

    sampling_probs: list
        Probabilities for sampling from the dataloaders

    batch_size: int
        Determines len of output sample list
    
    buffer_size: int
        The number of samples to load into buffer before shuffling

    mode: str
        either 'train' or 'validation'
    '''
    def __init__(self, dataloaders, sampling_probs, batch_size, mode):
        if len(dataloaders) != len(sampling_probs):
            raise ValueError('Number of dataloaders does not match sampling_probs')
        if np.sum(sampling_probs) != 1.0:
            raise ValueError(f'Sampling_probs sum to {np.sum(sampling_probs)} != 1.0')
        
        self.dataloaders = dataloaders
        self.batch_size = batch_size
        self.sampling_probs = sampling_probs
        self.dataloader_iterables = [iter(dataloader) for dataloader in dataloaders]
        self.dataloader_queues = [[] for dataloader in dataloaders]
        self.mode = mode
        
    def get_item(self, idx):
        try:
            return next(self.dataloader_iterables[idx])
        except StopIteration:
            if self.mode == 'train':
                self.dataloader_iterables[idx] = iter(self.dataloaders[idx])
                return next(self.dataloader_iterables[idx])
            else:
                raise StopIteration
 
    def __iter__(self):
        return self
    
    def __next__(self):
        datasource = np.random.multinomial(1, self.sampling_probs, size=self.batch_size) # batchsize x sampling_probs
        indexes = np.argmax(datasource, axis=1) # batchsize
        batch = []
        for idx in indexes:
            batch.append(self.get_item(idx))
        if self.batch_size == 1:
            return batch.pop(0)
        else:
            return batch
    
class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, img):
        device = img.device
        noise = torch.randn(img.shape).to(device) * self.sigma
        return img + noise

def color_augmentations(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = torch.nn.Sequential(
        RandomApply([color_jitter,
                    GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.2,0.2))], p=0.5),
        GaussianNoise(0.03)
        )
    return data_transforms

# def color_augmentations(size, s=0.5):
#     # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
#     """Return a set of data augmentation transformations as described in the SimCLR paper."""
#     color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     data_transforms = torch.nn.Sequential(
#         RandomApply([color_jitter,
#                     GaussianBlur(kernel_size=int(0.01 * size), sigma=(0.2,0.2))], p=0.5),
#         transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
#         transforms.RandomAutocontrast(p=0.5),
#         GaussianNoise(0.03)
#         )
#     return data_transforms

def center_crop(t, croph, cropw):
    _,_,h,w = t.shape
    startw = w//2-(cropw//2)
    starth = h//2-(croph//2)
    return t[:,:,starth:starth+croph,startw:startw+cropw]

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
        
def pad_up_to(array, crop_size, mode='constant'):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :return: padded array
    """
    xx, yy = crop_size
    c,h,w = array.shape

    t = (xx - h) // 2
    b = xx - t - h

    l = (yy - w) // 2
    r = yy - l - w
    t,b,l,r = [np.abs(i) for i in [t,b,l,r]]
    return np.pad(array, pad_width=((0, 0),(t, b), (l, r)), mode=mode)

def make_random_slice(img_shape, crop_size):
    h, w = img_shape
    crop_h, crop_w = crop_size
    if w-crop_w > 0:
        start_w = np.random.randint(0,w-crop_w)
    else:
        start_w = 0
    if h-crop_h > 0:
        start_h = np.random.randint(0,h-crop_h)
    else:
        start_h = 0
    return slice(start_h,start_h+crop_h), slice(start_w, start_w+crop_w)

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

class CropDataset(Dataset):
    def __init__(self, paths, raw_keys, label_keys, reader, crop_size=[512,512]):
        """
        paths: list of str
        raw_keys: list of str
        label_keys: list of str
        crop_size: list of int
        reader: 'h5py' or 'zarr'
        """
        self.file_paths = paths
        self.raw_keys = raw_keys
        self.label_keys = label_keys
        self.crop_size = crop_size
        if reader == 'h5py':
            self.reader = lambda x: h5py.File(x,'r')
        elif reader == 'zarr':
            self.reader = lambda x: zarr.open(x, 'r')
        else:
            if '.zarr' in self.file_paths[0]:
                self.reader = lambda x: zarr.open(x, 'r')
            elif '.hdf' in self.file_paths[0] or '.h5' in self.file_paths[0]:
                self.reader = lambda x: h5py.File(x,'r')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        with self.reader(self.file_paths[idx]) as f:
            if self.crop_size:
                slice_h, slice_w = make_random_slice(f[self.raw_keys[0]].shape[-2:], self.crop_size)
            raws = [np.array(f[key][...,slice_h, slice_w]) for key in self.raw_keys]
            labels = [np.array(f[key][...,slice_h, slice_w]) for key in self.label_keys]
        raws = [normalize_percentile(raw.astype(np.float32)) for raw in raws]
        if self.crop_size:
            for i in range(len(raws)):
                if raws[i].shape[-2]<self.crop_size[0] or raws[i].shape[-1]<self.crop_size[1]:
                    raws[i] = pad_up_to(raws[i], self.crop_size)
            if labels:
                for i in range(len(labels)):
                    if labels[i].shape[-2]<self.crop_size[0] or labels[i].shape[-1]<self.crop_size[1]:
                        labels[i] = pad_up_to(labels[i], self.crop_size)
        if labels:
            return *raws, *labels
        else:
            return *raws, False

def shuffle_train_data(X_train, Y_train, random_seed):
    """
    Shuffles data with seed 1.
    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train : array(float)
        shuffled array of training images.
    Y_train : array(float)
        Shuffled array of labelled training images.
    """
    np.random.seed(random_seed)
    seed_ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[seed_ind]
    Y_train = Y_train[seed_ind]

    return X_train, Y_train

def zero_out_train_data(X_train, Y_train, fraction):
    """
    Fractionates training data according to the specified `fraction`.
    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    fraction: float (between 0 and 100)
        fraction of training images.
    Returns
    -------
    X_train : array(float)
        Fractionated array of source images.
    Y_train : float
        Fractionated array of label images.
    """
    train_frac = int(np.round((fraction / 100) * X_train.shape[0]))
    Y_train[train_frac:] *= 0

    return X_train, Y_train


def convert_to_oneHot(data, eps=1e-8):
    """
    Converts labelled images (`data`) to one-hot encoding.
    Parameters
    ----------
    data : array(int)
        Array of lablelled images.
    Returns
    -------
    data_oneHot : array(int)
        Array of one-hot encoded images.
    """
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
        if ( np.abs(np.max(data[i])) <= eps ):
            data_oneHot[i][...,0] *= 0

    return data_oneHot


def add_boundary_label(lbl, dtype=np.uint16):
    """
    Find boundary labels for a labelled image.
    Parameters
    ----------
    lbl : array(int)
         lbl is an integer label image (not binarized).
    Returns
    -------
    res : array(int)
        res is an integer label image with boundary encoded as 2.
    """

    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res

def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot

def get_closest_to_median(crd, all):
    sq_dist = (all**2).sum(0) + crd.dot(crd) - 2*crd.dot(all)
    return all[:,int(sq_dist.argmin())]

def inst_to_3c_cp(gt_labels, sigma = 2):
    '''
    Generate 3class map and center of masses for each instance
    '''
    gt_labels = np.squeeze(gt_labels)
    gt_threeclass = np.zeros(gt_labels.shape, dtype=np.uint8)
    gt_centers = np.zeros(gt_labels.shape, dtype=np.float32)
    struct = generate_binary_structure(2,2) # TODO can replace this with a np.full([3,3], True)
    crds = np.array(np.meshgrid(np.arange(gt_labels.shape[1]), np.arange(gt_labels.shape[0]))).transpose(1,2,0)
    gt_coords = []
    for inst in np.unique(gt_labels):
        if inst == 0:
            continue
        lab = gt_labels == inst
        tmp = crds[lab]
        y,x = np.median(tmp, axis=0).astype(np.int32) #Take median instead of mean of coordinates
        if gt_labels[x,y] == 0:
            y,x = get_closest_to_median(np.array([y,x]), tmp.T)
        # TODO we never check if this point is actually in the cell, however I would argue that 

        # x,y = center_of_mass(lab) 
        eroded_lab = binary_erosion(lab, iterations=1, structure=struct, border_value=1)
        boundary = np.logical_xor(lab, eroded_lab)
        gt_threeclass[boundary] = 2
        gt_threeclass[eroded_lab] = 1
        gt_centers[int(x),int(y)]  = 1.
        gt_coords.append([inst,x,y])
    # gt_centers = gaussian_filter(gt_centers, sigma, mode='constant')
    # max_value = np.max(gt_centers)
    # if max_value >0:
    #     gt_centers /= max_value # rescale to [0,1]
    # gt_centers = gt_centers.astype(np.float32)
    return gt_threeclass[np.newaxis,], gt_centers[np.newaxis,], np.array(gt_coords)[np.newaxis,]

def add_3c_gt(Y):
    instances = Y[...,0]
    gt_3c_list = []
    for inst in tqdm(instances):
        gt_3c,_,_ = inst_to_3c_cp(inst)
        gt_3c_list.append(gt_3c)
    gt_3c = np.transpose(np.stack(gt_3c_list, 0), [0,2,3,1])
    Y = np.concatenate([Y, gt_3c],-1)
    return Y