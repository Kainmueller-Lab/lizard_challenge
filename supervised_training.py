import os
os.environ["OMP_NUM_THREADS"]="1"
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
import toml
from time import time
import segmentation_models_pytorch as smp
from torchvision.transforms.transforms import RandomApply, GaussianBlur, ColorJitter
from src.data_utils import GaussianNoise
from src.color_conversion import Rgb2Hed, Hed2Rgb, LinearContrast
from src.embedding_loss import SpatialEmbLoss
from src.focal_loss import FocalCE
from src.unet import UNet
from src.multi_head_unet import UnetDecoder, MultiHeadModel
from src.spatial_augmenter import SpatialAugmenter
from src.train_utils import supervised_train_step, validation, save_snapshot, save_model, make_cpvs
from src.data_utils import SliceDataset, CropDataset, color_augmentations, add_3c_gt
#import torch_optimizer as optim

torch.backends.cudnn.benchmark = True

torch.manual_seed(42)

params = {
    'data_path': 'tiles',
    'experiment' : 'uniform_MH_attention',
    'batch_size': 8,
    'training_steps':400000,
    'in_channels': 3,
    'num_fmaps': 32,
    'fmap_inc_factors': 2,
    'downsample_factors': [ [ 2, 2,], [ 2, 2,], [ 2, 2,], [ 2, 2,],],
    'num_fmaps_out': 12,
    'constant_upsample': False,
    'padding': 'same',
    'activation': 'ReLU',
    'weight_decay': 1e-4,
    'learning_rate': 5e-4,
    'seed': 42,
    'num_validation': 500,
    'checkpoint_path': None, # 'exp_0_dsb/best_model',
    'pretrained_model': True,
    'multi_head': True,
    'uniform_class_sampling': False,
    'optimizer': 'AdamW', # one of SGD AdamW AdaBound , Adahessian breaks memory and is not supported
    'validation_step' : 500,
    'snapshot_step' : 5000,
    'checkpoint_step': 20000,
    'instance_seg': 'cpv_3c', # 'embedding' or 'cpv_3c'
    'attention': True
    }

aug_params_fast = {
    'mirror': {'prob_x': 0.5, 'prob_y': 0.5, 'prob': 0.5},
    'translate': {'max_percent':0.05, 'prob': 0.2},
    'scale': {'min': 0.8, 'max':1.2, 'prob': 0.2},
    'zoom': {'min': 0.8, 'max':1.2, 'prob': 0.2},
    'rotate': {'max_degree': 179, 'prob': 0.75},
    'shear': {'max_percent': 0.1, 'prob': 0.2},
    'elastic': {'alpha': [120,120], 'sigma': 8, 'prob': 0.5}
}

log_dir = os.path.join(params['experiment'],'train')
snap_dir = os.path.join(log_dir,'snaps')
os.makedirs(snap_dir,exist_ok=True)
writer_dir = os.path.join(log_dir,'summary',str(time()))
os.makedirs(writer_dir,exist_ok=True)
writer = SummaryWriter(writer_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
params['device'] = device
params['aug_params_fast'] = aug_params_fast
with open(os.path.join(params['experiment'], 'params.toml'), 'w') as f:
    toml.dump(params, f)

if 'tiles' in params['data_path']:
    img_data =  np.load(os.path.join(params['data_path'], 'images.npy')) # N x H x W x C
    lab_dat = np.load(os.path.join(params['data_path'], 'labels.npy')) # N x H x W x 2
    np.random.seed(params['seed'])
    seed_ind = np.random.permutation(img_data.shape[0])
    img_data = img_data[seed_ind]
    lab_dat = lab_dat[seed_ind]
    X_train = img_data[:-params['num_validation']]
    Y_train = lab_dat[:-params['num_validation']]
    X_val = img_data[-params['num_validation']:]
    Y_val = lab_dat[-params['num_validation']:]
    # add gt_3c
    if params['instance_seg'] == 'cpv_3c':
        print('Add 3c labels')
        Y_train = add_3c_gt(Y_train)
        Y_val = add_3c_gt(Y_val)
    labeled_dataset = SliceDataset(raw=X_train, labels=Y_train)
    validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
else:
    file_paths = [os.path.join(params['data_path'], file) for file in os.listdir(params['data_path'])]
    random.Random(params['seed']).shuffle(file_paths)
    val_paths = [file_paths.pop(0) for _ in range(params['num_validation'])]
    labeled_paths = file_paths
    labeled_dataset = CropDataset(labeled_paths, raw_keys=['raw'], label_keys=['gt_labels', 'gt_instances'], crop_size=[256,256])
    validation_dataset = CropDataset(val_paths, raw_keys=['raw'], label_keys=['gt_labels', 'gt_instances'], crop_size=[256,256])

if params['uniform_class_sampling']:
    print('Uniform class sampling')
    # uniform class sampling
    labeled_dataloader = DataLoader(labeled_dataset,
                        batch_size=1,
                        shuffle=False,
                        prefetch_factor=4,
                        num_workers=6)
    classes = [0,1,2,3,4,5,6]
    count_list = []

    for raw, gt in labeled_dataloader:
        gt_classes = gt[...,1].squeeze()
        tmp_list = []
        for c in classes:
            tmp_list.append(torch.sum(gt_classes==c)) # sum of individual classes for a sample
        count_list.append(torch.stack(tmp_list)) # 
        
    counts = torch.stack(count_list) # n_samples x classes 
    sampling_weights = counts/counts.sum(0).unsqueeze(0) # n_samples x classes / 1 x classes = n_samples x classes
    sampling_weights = sampling_weights.sum(1) # n_samples
    sampler = torch.utils.data.WeightedRandomSampler(sampling_weights, num_samples=len(sampling_weights), replacement=True)

    labeled_dataloader = DataLoader(labeled_dataset,
                        batch_size=1,
                        prefetch_factor=4,
                        sampler=sampler,
                        num_workers=6)
else:
    labeled_dataloader = DataLoader(labeled_dataset,
                        batch_size=params['batch_size'],
                        shuffle=True,
                        prefetch_factor=4,
                        num_workers=6)

validation_dataloader = DataLoader(validation_dataset,
                    batch_size=1,
                    shuffle=True,
                    prefetch_factor=4,
                    num_workers=1)

if params['pretrained_model']:
    if params['multi_head']:
        encoder = smp.encoders.get_encoder(
            name= "timm-efficientnet-b5",
            in_channels=3,
            depth=5,
            weights="imagenet").to(device)
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_inst = UnetDecoder(
                    encoder_channels=encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_batchnorm=True,
                    center=False,
                    attention_type='scse' if params['attention'] else None
                    ).to(device)
        decoder_ct = UnetDecoder(
                    encoder_channels=encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_batchnorm=True,
                    center=False,
                    attention_type='scse' if params['attention'] else None
                    ).to(device)
        head_inst = smp.base.SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=5,
                    activation=None,
                    kernel_size=1).to(device)
        head_ct = smp.base.SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=7,
                    activation=None,
                    kernel_size=1).to(device)

        decoders = [decoder_inst, decoder_ct]
        heads = [head_inst, head_ct]
        model = MultiHeadModel(encoder, decoders, heads)
    else:
        model = smp.Unet(
            encoder_name= "timm-efficientnet-b5", # "timm-efficientnet-b5", # choose encoder
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['num_fmaps_out'],                      # model output channels (number of classes in your dataset)
            ).to(params['device'])
#preprocess_input = smp.encoders.get_preprocessing_fn('efficientnet-b5', pretrained='imagenet')
else:
    model = UNet(in_channels = params['in_channels'],
                num_fmaps = params['num_fmaps'],
                fmap_inc_factor = params['fmap_inc_factors'],
                downsample_factors = params['downsample_factors'],
                activation = params['activation'],
                padding = params['padding'],
                num_fmaps_out = params['num_fmaps_out'],
                constant_upsample = params['constant_upsample'],
            ).to(params['device'])

if 'checkpoint_path' in params.keys() and params['checkpoint_path']:
    model.load_state_dict(torch.load(params['checkpoint_path'])['model_state_dict'])
model = model.train()

if 'cutout_or_RandomErasing' in params.keys() and params['cutout_or_RandomErasing'] == 'RandomErasing':
    eraser = torch.nn.Sequential(
        *[transforms.RandomErasing(p=1., scale=(0.001, 0.001), ratio=(0.3,1.3), inplace=False) for _ in range(200)])
    def cutout(img):
        B,C,_,_ = img.shape
        device = img.device
        img_list = []
        mask_list = []
        for b in range(B):
            mask = eraser(torch.ones_like(img[b:b+1,...]).to(device))
            img_masked = img[b:b+1,...] * mask.float()
            mask = -1 * (mask-1)
            img_list.append(img_masked) 
            mask_list.append(mask)
        return torch.cat(img_list, axis=0), torch.cat(mask_list, axis=0)

# initialize optimizer, augmentations and summary writer
if params['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params['learning_rate'],
                                momentum=0.9,
                                weight_decay=params['weight_decay'],
                                nesterov=True)
elif params['optimizer'] == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['learning_rate'],
                                  weight_decay=params['weight_decay'])

# elif params['optimizer'] == 'AdaBound':
#     optimizer = optim.AdaBound(model.parameters(),
#                                lr=params['learning_rate'],
#                                final_lr = 1e-5,
#                                weight_decay=params['weight_decay'])

#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['training_steps'], eta_min=1e-5)
fast_aug = SpatialAugmenter(aug_params_fast)#, padding_mode='reflection')
if params['instance_seg'] == 'embedding':
    inst_loss_fn = SpatialEmbLoss(n_sigma=2, to_center=False, foreground_weight=10, H=256,W=256).to(device)
elif params['instance_seg'] == 'cpv_3c':
    def inst_loss_fn(input, gt_inst, gt_3c):
        gt_cpv_list = []
        gt_inst = gt_inst.squeeze(0)
        for b in range(gt_inst.shape[0]):
            gt_cpv_list.append(make_cpvs(gt_inst.to(device)[b], device))
        gt_cpv = torch.cat(gt_cpv_list, axis=0)
        loss_cpv = F.mse_loss(input = input[:,:2],target = gt_cpv.float())
        loss_3c =  F.cross_entropy(input = input[:,2:],
                        target = gt_3c.long().squeeze(0).to(device),
                        weight=torch.tensor([1,1,2]).float().to(device))
        return loss_cpv + loss_3c

ce_loss_fn = FocalCE(num_classes=7)

def color_augmentations(size, s=0.2):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.0 * s, 0.8 * s, 0.2 * s) # brightness, contrast, saturation, hue
    HED_contrast = torch.nn.Sequential(
                Rgb2Hed(),
                LinearContrast(alpha=(0.75,1.25)),
                Hed2Rgb())
    data_transforms = torch.nn.Sequential(
        RandomApply([
            HED_contrast,
            color_jitter,
            GaussianNoise(0.005), 
            GaussianBlur(kernel_size=3, sigma=(0.1,0.1))], p=0.5),
        )
    return data_transforms

color_aug_fn = color_augmentations(200, s=0.2)
validation_loss = []
step = -1

def supervised_training(params):
    step = -1
    while step < params['training_steps']:
        tmp_loader = iter(labeled_dataloader)
        for raw, gt in tmp_loader:
            step += 1
            optimizer.zero_grad()
            loss, pred_inst, pred_class ,img_caug, gt_inst, gt_ct, cutout_maps = supervised_train_step(model,
                                                                raw,
                                                                gt,
                                                                fast_aug,
                                                                color_aug_fn,
                                                                cutout,
                                                                params['cutout_prob'],
                                                                inst_loss_fn,
                                                                ce_loss_fn,
                                                                writer,
                                                                device,
                                                                step)
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            if step % params['validation_step'] == 0:
                val_new = validation(model,
                        validation_dataloader,
                        inst_loss_fn,
                        ce_loss_fn,
                        device,
                        step,
                        writer)
                validation_loss.append(val_new)
                if val_new <= np.min(validation_loss):
                    print('Save best model')
                    save_model(step, model, optimizer, loss, os.path.join(log_dir,"best_model"))
            #
            if step % params['snapshot_step'] == 0:
                print('Save snapshot')
                tmp_dic = {
                        'pred_ct': pred_class[0].cpu().detach().numpy(),
                        'img_caug': img_caug[0].squeeze(0).cpu().detach().numpy(),
                        'gt_inst': gt_inst.squeeze(0).cpu().detach().numpy()[:1],
                        'gt_ct': gt_ct.squeeze(0).cpu().detach().numpy()[:1]}
                if params['instance_seg'] == 'embedding':
                    _,_,h,w = pred_inst.shape
                    xym_s = inst_loss_fn.xym[:, 0:h, 0:w].contiguous()
                    spatial_emb = pred_inst[0, 0:2] + xym_s  # 2 x h x w
                    sigma = pred_inst[0, 2:2+inst_loss_fn.n_sigma]  # n_sigma x h x w
                    seed_map = torch.sigmoid(
                        pred_inst[0, 2+inst_loss_fn.n_sigma:2+inst_loss_fn.n_sigma + 1])  # 1 x h x w
                    tmp_dic['embedding'] = spatial_emb.cpu().detach().numpy()
                    tmp_dic['sigma'] = sigma.cpu().detach().numpy()
                    tmp_dic['seed_map'] = seed_map.cpu().detach().numpy()
                elif params['instance_seg'] == 'cpv_3c':
                    tmp_dic['pred_cpv'] = pred_inst[0,:2].cpu().detach().numpy()
                    tmp_dic['pred_3c'] = pred_inst[0,2:].softmax(0).cpu().detach().numpy()
                save_snapshot(snap_dir, tmp_dic, step)
            if step % params['checkpoint_step'] == 0:
                save_model(step, model, optimizer, loss, os.path.join(log_dir,"checkpoint_step_"+str(step)))    


#if __name__ == '__main__':
supervised_training(params)