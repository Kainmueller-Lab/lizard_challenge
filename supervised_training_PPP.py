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
from src.train_utils import save_snapshot, save_model, make_cpvs
from src.data_utils import SliceDataset, CropDataset, color_augmentations, add_3c_gt
from src.disc_loss import DiscriminativeLoss
from src.data_utils import center_crop

#import torch_optimizer as optim

torch.backends.cudnn.benchmark = True

torch.manual_seed(42)

params = {
    'data_path': '/fast/AG_Kainmueller/jrumber/data/lizard/tiles',
    'experiment' : 'uniform_MH_big_ema_PPP_val50',#'uniform_MH_3c_big_ema_PPP',
    'batch_size': 1,
    'training_steps':200000,
    'in_channels': 3,
    'num_fmaps': 32,
    'fmap_inc_factors': 2,
    'downsample_factors': [ [ 2, 2,], [ 2, 2,], [ 2, 2,], [ 2, 2,],],
    'num_fmaps_out': 12,
    'constant_upsample': False,
    'padding': 'same',
    'activation': 'ReLU',
    'weight_decay': 1e-4,
    'learning_rate': 0.5e-3,
    'seed': 42,
    'num_validation': 50,
    'cutout_prob':0.0,
    'checkpoint_path': 'uniform_MH_3c_big_ema/train/best_model',
    'cutout_or_RandomErasing': 'RandomErasing',
    'pretrained_model': True,
    'multi_head': True,
    'uniform_class_sampling': True,
    'optimizer': 'AdamW', # one of SGD AdamW AdaBound
    'validation_step' : 1000,
    'snapshot_step' : 5000,
    'checkpoint_step': 20000,
    'instance_seg': 'PPP',
    'inst_channels': 290,# 5 
    'attention': False,
    'ema_loss': True,
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
            name= "timm-efficientnet-b7",
            in_channels=3,
            depth=5,
            weights="noisy-student").to(device)
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
                    out_channels= 5,
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
            encoder_name= "timm-efficientnet-b7", # "timm-efficientnet-b5", # choose encoder
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
    print('Load checkpoint step', str(torch.load(params['checkpoint_path'])['step']))

if params['instance_seg'] == 'PPP':
    # replace inst decoder and head with PPP head
    decoder_inst_PPP = UnetDecoder(
                    encoder_channels=encoder.out_channels,
                    decoder_channels=(256, 128, 128, 128, 128),
                    n_blocks=5,
                    use_batchnorm=True,
                    center=False,
                    attention_type='scse' if params['attention'] else None
                    ).to(device)
    head_inst_PPP = smp.base.SegmentationHead(
                in_channels = 128,
                out_channels = 290,
                activation=None,
                kernel_size=1).to(device)
    
    model.decoders[0] = decoder_inst_PPP
    model.heads[0] = head_inst_PPP

model = model.train()

def supervised_train_step(model, raw, gt, fast_aug, color_aug_fn, inst_lossfn, class_lossfn, writer, device, step, inst_channels):
    raw = raw.to(device).float()
    raw = raw + raw.min() *-1
    raw /= raw.max()
    gt = gt.to(device).float()
    B,_,_,_ = raw.shape
    if gt.shape[-1]>2:
        cpv_3c_model = True
        gt_3c_list = []
    else:
        cpv_3c_model = False
    raw_list = []
    gt_inst_list = []
    gt_ct_list = []
    cutout_maps = []
    for b in range(B):
        img = raw[b].permute(2,0,1).unsqueeze(0) # BHWC -> BCHW
        gt_ = gt[b].permute(2,0,1).unsqueeze(0) # BHW2 -> B2HW
        img_saug, gt_saug = fast_aug.forward_transform(img, gt_)                
        #gt_inst = fix_mirror_padding(gt_saug[0,0].cpu().detach().numpy().astype(np.int32)) # slow af
        #gt_inst = torch.tensor(gt_inst, device=device).float().unsqueeze(0)
        gt_inst = gt_saug[:,0]
        gt_ct = gt_saug[:,1]
        img_caug = color_aug_fn(img_saug)
        #
        raw_list.append(img_caug)
        gt_inst_list.append(gt_inst)
        gt_ct_list.append(gt_ct)
    img_caug = torch.cat(raw_list, axis = 0)
    gt_inst  = torch.cat(gt_inst_list, axis = 0)
    gt_ct = torch.cat(gt_ct_list, axis = 0)
    out_fast = model(img_caug)
    _,_,H,W = out_fast.shape
    gt_inst = center_crop(gt_inst.unsqueeze(0), H, W)
    gt_ct = center_crop(gt_ct.unsqueeze(0), H, W)
    pred_inst = out_fast[:,:inst_channels]
    pred_class = out_fast[:,inst_channels:]
    #
    instance_loss = inst_lossfn(pred_inst, gt_inst.squeeze(0).float(), (gt_inst.squeeze(0)>0).float())
    class_loss = class_lossfn(pred_class, gt_ct.long())
    if torch.isnan(instance_loss) or not torch.isfinite(instance_loss):
        instance_loss = torch.tensor(0.0)
    if torch.isnan(class_loss) or not torch.isfinite(class_loss):
        class_loss = torch.tensor(0.0)
    loss = instance_loss + class_loss
    writer.add_scalar('instance_loss', instance_loss, step)
    writer.add_scalar('class_loss', class_loss, step)
    writer.add_scalar('loss', loss, step)
    print('loss', loss.item())
    return loss, pred_inst, pred_class.softmax(1),img_caug, gt_inst, gt_ct

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

def make_affinities(gt_inst, patchshape = [1, 17, 17]):
    nhood = []
    patchshape = [1, 17, 17]
    psH = np.array(patchshape)//2
    for i in range(-psH[1], psH[1]+1):
        for j in range(-psH[2], psH[2]+1):
            nhood.append([i,j])

    # nhood = torch.Tensor(nhood, dtype=torch.int32)
    nhood = torch.cuda.IntTensor(nhood)

    nEdge = nhood.size()[0]
    dims = nhood.size()[1]
    seg = gt_inst
    shape = seg.size()

    aff_shape = [int(l) for l in list((nEdge,) + tuple(seg.shape))]
    aff = torch.cuda.IntTensor(*aff_shape).fill_(0)
    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                     max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                 seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                     * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                             max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                             * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )
    return aff

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['training_steps'], eta_min=1e-5)
fast_aug = SpatialAugmenter(aug_params_fast) #, padding_mode='reflection')
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
elif params['instance_seg'] == 'disc_loss':
    disc_loss = DiscriminativeLoss().to(device)
    def inst_loss_fn(input, gt_inst, gt_fgbg):
        B,_,_,_ = input.shape
        loss = 0
        for b in range(0,B):
            d_loss,_ = disc_loss(input[b:b+1,1:], gt_inst[b:b+1].unsqueeze(0))
            fgbg_loss = F.binary_cross_entropy_with_logits(input=input[b:b+1,0], target=gt_fgbg[b:b+1])
            loss += d_loss/10.0 + fgbg_loss
        return loss

elif params['instance_seg'] == 'PPP':
    def inst_loss_fn(input, gt_inst, gt_fgbg):
        gt_affinities = make_affinities(gt_inst.squeeze(0)).unsqueeze(0)
        fgbg_loss = F.binary_cross_entropy(input=input[:,0,...].sigmoid(), target=gt_fgbg.float())
        affinity_loss = F.binary_cross_entropy(input=input[:,1:,...].squeeze().sigmoid(), target=gt_affinities.squeeze().float())
        loss = fgbg_loss + affinity_loss
        loss *= 10.0
        return loss

ce_loss_fn = FocalCE(num_classes=7, ema=params['ema_loss'])

def color_augmentations(size, s=0.2):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.0 * s, 0.8 * s, 0.2 * s) # brightness, contrast, saturation, hue
    HED_contrast = torch.nn.Sequential(
                Rgb2Hed(),
                LinearContrast(alpha=(0.65,1.35)),
                Hed2Rgb())
    data_transforms = torch.nn.Sequential(
        RandomApply([
            HED_contrast,
            color_jitter,
            GaussianNoise(0.005), 
            GaussianBlur(kernel_size=3, sigma=(0.1,0.1))], p=0.5),
        )
    return data_transforms

def validation(model, validation_dataloader, inst_lossfn, class_lossfn, device, step, writer, inst_channels):
    val_loss = []
    val_inst_loss = []
    val_ct_loss = []
    for raw, gt in validation_dataloader:
        raw = raw.to(device)
        raw = raw.float() + raw.min() *-1
        raw /= raw.max()
        gt = gt.to(device)
        raw = raw.permute(0,3,1,2) # BHWC -> BCHW
        gt = gt.permute(0,3,1,2) # BHW2 -> B2HW
        with torch.no_grad():
            out = model(raw)
            b,c,h,w = out.shape
            gt = center_crop(gt, h, w)
            gt_inst = gt[:,0]
            gt_ct = gt[:,1]
            pred_inst = out[:,:inst_channels]
            pred_class = out[:,inst_channels:]
            instance_loss = inst_lossfn(pred_inst, gt_inst.float(), (gt_inst>0).float())
            class_loss = class_lossfn(pred_class, gt_ct.unsqueeze(0).long())
            loss = instance_loss + class_loss
            val_loss.append(loss.item())
            val_inst_loss.append(instance_loss.item())
            val_ct_loss.append(class_loss.item())
        val_new = np.mean(val_loss)
        inst_new = np.mean(val_inst_loss)
        ct_new = np.mean(val_ct_loss)
        writer.add_scalar('val_loss', val_new, step)
        writer.add_scalar('val_inst', inst_new, step)
        writer.add_scalar('val_ct', ct_new, step)
        print('Validation loss: ', val_new)
        return val_new

color_aug_fn = color_augmentations(200, s=0.4)
validation_loss = []
step = -1

def supervised_training(params):
    step = -1
    while step < params['training_steps']:
        tmp_loader = iter(labeled_dataloader)
        for raw, gt in tmp_loader:
            step += 1
            optimizer.zero_grad()
            loss, pred_inst, pred_class ,img_caug, gt_inst, gt_ct = supervised_train_step(model,
                                                                raw,
                                                                gt,
                                                                fast_aug,
                                                                color_aug_fn,
                                                                inst_loss_fn,
                                                                ce_loss_fn,
                                                                writer,
                                                                device,
                                                                step,
                                                                inst_channels=params['inst_channels'])
            if torch.isnan(loss) or not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if step % params['validation_step'] == 0:
                val_new = validation(model,
                        validation_dataloader,
                        inst_loss_fn,
                        ce_loss_fn,
                        device,
                        step,
                        writer,
                        inst_channels=params['inst_channels'])
                validation_loss.append(val_new)
                if val_new >= np.max(validation_loss):
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
                elif params['instance_seg'] == 'PPP':
                    tmp_dic['pred_fgbg'] = pred_inst[0,:1].sigmoid().cpu().detach().numpy()
                    tmp_dic['pred_aff'] = pred_inst[0,1:21].sigmoid().cpu().detach().numpy() 
                save_snapshot(snap_dir, tmp_dic, step)
            if step % params['checkpoint_step'] == 0:
                save_model(step, model, optimizer, loss, os.path.join(log_dir,"checkpoint_step_"+str(step)))    


if __name__ == '__main__':
    supervised_training(params)