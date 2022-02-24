import torch
import torch.nn.functional as F
from .data_utils import center_crop
import h5py
import os
import numpy as np
from scipy.ndimage import measurements

from .stain_mix import *

def save_model(step, model, optimizer, loss, filename):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, filename)

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
                out = out[...,4:-4,4:-4]
            mask = torch.ones_like(out[:,-1:,:,:])
            slow_aug.interpolation='nearest'
            out_inv, aug_mask_inv = slow_aug.inverse_transform(out, mask)
            tmp_out_list.append(out_inv*aug_mask_inv)
            tmp_mask_list.append(aug_mask_inv)
        out_slow = torch.stack(tmp_out_list).sum(0) # 1 x 3 x H x W
        mask_slow = torch.stack(tmp_mask_list).sum(0) > 0 # 1 x 1 x H x W
        n_out = torch.stack(tmp_mask_list).sum(0)
        out_slow = mask_slow*out_slow/(n_out+1e-6)
        out_slow = F.pad(out_slow, (4,4,4,4))
        mask_slow = F.pad(mask_slow, (4,4,4,4))
        out_list.append(out_slow)
        mask_list.append(mask_slow)
    out = torch.cat(out_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    return out, mask


def make_pseudolabel_wstain(raw, model, n_views, slow_aug, source_stain, target_stain):
    B,C,H,w = raw.shape
    mask_list = []
    out_list = []
    for b in range(B): 
        tmp_out_list = []
        tmp_mask_list = []
        for _ in range(n_views):
            # gen views
            slow_aug.interpolation='bilinear'
            raw = torch_stain_mixup(raw[b].unsqueeze(0), source_stain, target_stain, intensity_range=[0.95,1.05], alpha=.6)           
            view = slow_aug.forward_transform(raw[b].unsqueeze(0))
            with torch.no_grad():
                out = model(view)
                out = out[...,4:-4,4:-4]
            mask = torch.ones_like(out[:,-1:,:,:])
            slow_aug.interpolation='nearest'
            out_inv, aug_mask_inv = slow_aug.inverse_transform(out, mask)
            tmp_out_list.append(out_inv*aug_mask_inv)
            tmp_mask_list.append(aug_mask_inv)
        out_slow = torch.stack(tmp_out_list).sum(0) # 1 x 3 x H x W
        mask_slow = torch.stack(tmp_mask_list).sum(0) > 0 # 1 x 1 x H x W
        n_out = torch.stack(tmp_mask_list).sum(0)
        out_slow = mask_slow*out_slow/(n_out+1e-6)
        out_slow = F.pad(out_slow, (4,4,4,4))
        mask_slow = F.pad(mask_slow, (4,4,4,4))
        out_list.append(out_slow)
        mask_list.append(mask_slow)
    out = torch.cat(out_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    return out, mask


def supervised_train_step(model, raw, gt, fast_aug, color_aug_fn, cutout_fn, cutout_prob, inst_lossfn, class_lossfn, writer, device, step):
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
        if cpv_3c_model:
            gt_3c = gt_saug[:,2]
            gt_3c_list.append(gt_3c)
        img_caug = color_aug_fn(img_saug)
        if torch.rand(1)<cutout_prob:
            img_caug, cutout_map = cutout_fn(img_caug)
            no_cutout=False
        else:
            no_cutout = True
            cutout_map = torch.zeros_like(img_caug)
        raw_list.append(img_caug)
        gt_inst_list.append(gt_inst)
        gt_ct_list.append(gt_ct)
        cutout_maps.append(cutout_map)
    img_caug = torch.cat(raw_list, axis = 0)
    gt_inst  = torch.cat(gt_inst_list, axis = 0)
    gt_ct = torch.cat(gt_ct_list, axis = 0)
    cutout_maps = torch.cat(cutout_maps, axis = 0)
    out_fast = model(img_caug)
    _,_,H,W = out_fast.shape
    gt_inst = center_crop(gt_inst.unsqueeze(0), H, W)
    gt_ct = center_crop(gt_ct.unsqueeze(0), H, W)
    pred_inst = out_fast[:,:5]
    pred_class = out_fast[:,5:]
    if cpv_3c_model:
        gt_3c = torch.cat(gt_3c_list, axis = 0)
        gt_3c = center_crop(gt_3c.unsqueeze(0), H, W)
        instance_loss = inst_lossfn(pred_inst, gt_inst, gt_3c)
    else:
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
#     if not no_cutout:
#         inpaint_loss = torch.masked_select(loss_img, cutout_maps.any(1).bool()).mean()
#         writer.add_scalar('sup_inpainting_loss', inpaint_loss.item(), step)
#         print('emb inpaint_loss', inpaint_loss.item())
#         loss += inpaint_loss
    return loss, pred_inst, pred_class.softmax(1),img_caug, gt_inst, gt_ct, cutout_maps

def semantic_seg_train_step(model, raw, gt, fast_aug, color_aug_fn, class_lossfn, writer, device, step):
    raw = raw.to(device).float()
    raw = raw + raw.min() *-1
    raw /= raw.max()
    gt = gt.to(device).float()
    B,_,_,_ = raw.shape
    raw_list = []
    gt_ct_list = []
    for b in range(B):
        img = raw[b].permute(2,0,1).unsqueeze(0) # BHWC -> BCHW
        gt_ = gt[b].permute(2,0,1).unsqueeze(0) # BHW2 -> B2HW
        img_saug, gt_saug = fast_aug.forward_transform(img, gt_)                
        gt_ct = gt_saug[:,1]
        img_caug = color_aug_fn(img_saug)
        raw_list.append(img_caug)
        gt_ct_list.append(gt_ct)
    img_caug = torch.cat(raw_list, axis = 0)
    gt_ct = torch.cat(gt_ct_list, axis = 0)
    out_fast = model(img_caug)
    _,_,H,W = out_fast.shape
    gt_ct = center_crop(gt_ct.unsqueeze(0), H, W)
    pred_class = out_fast
    class_loss = class_lossfn(pred_class, gt_ct.long())
    if torch.isnan(class_loss) or not torch.isfinite(class_loss):
        class_loss = torch.tensor(0.0)
    class_loss = class_loss
    writer.add_scalar('class_loss', class_loss, step)
    print('loss', class_loss.item())
    return class_loss, pred_class.softmax(1),img_caug, gt_ct

def semantic_seg_validation(model, validation_dataloader, class_lossfn, device, step, writer):
    val_loss = []
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
            gt_ct = gt[:,1]
            pred_class = out
            class_loss = class_lossfn(pred_class, gt_ct.unsqueeze(0).long())
            val_loss.append(class_loss.item())
        val_new = np.mean(val_loss)
        writer.add_scalar('val_loss', val_new, step)
        print('Validation loss: ', val_new)
        return val_new

def save_snapshot(log_dir, out_dict, step):
    print('Save training snapshot')
    with h5py.File(os.path.join(log_dir,'snap_step_'+str(step)),'w') as f:
        for key in list(out_dict.keys()):
            f.create_dataset(key, data = out_dict[key].astype(np.float32))      

def validation(model, validation_dataloader, inst_lossfn, class_lossfn, device, step, writer):
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
            pred_inst = out[:,:5]
            pred_class = out[:,5:]
            if gt.shape[1]>2:
                gt_3c = gt[:,2]
                instance_loss = inst_lossfn(pred_inst, gt_inst.unsqueeze(0), gt_3c.unsqueeze(0))
            else:
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
    

def unsupervised_train_step(raw, model, slow_model, projector, n_views, fast_aug, slow_aug, color_aug_fn, cutout_prob, cutout_fn, semi_loss_fn, device, writer, step):
    raw = raw.to(device)
    raw = raw.float() + raw.min() *-1
    raw /= raw.max()
    pseudolabel, mask_slow = make_pseudolabel(raw, slow_model, n_views, slow_aug)
    if fast_aug:
        img_saug = fast_aug.forward_transform(raw)
        random_state = fast_aug.random_state.copy()
        pseudolabel_saug, mask_slow = fast_aug.forward_transform(pseudolabel, mask_slow.float(), random_state=random_state)
    else:
        img_saug = raw
        pseudolabel_saug, mask_slow = pseudolabel, mask_slow.float()
    img_caug = color_aug_fn(img_saug)
    if torch.rand(1)<cutout_prob:
        img_caug, cutout_map = cutout_fn(img_caug)
        no_cutout=False
    else:
        no_cutout = True
        cutout_map = torch.zeros_like(img_caug)
    out_fast = model(img_caug)
    out_fast = out_fast*mask_slow
    pseudolabel_saug = pseudolabel_saug*mask_slow
    # loss calculation
    out_fast_proj = projector(out_fast)
    pseudolabel_saug_proj = projector(pseudolabel_saug)
    loss_img = semi_loss_fn(input=out_fast_proj, target=pseudolabel_saug_proj.detach())
    loss_img *= mask_slow.squeeze(0)
    con_loss = loss_img.mean() 
    print('con loss', con_loss.item())
    writer.add_scalar('contrastive_loss', con_loss.item(), step)
    if no_cutout:
        loss = con_loss
    else:
        inpaint_loss = torch.masked_select(loss_img, cutout_map.any(1).bool()).mean()
        writer.add_scalar('con_inpainting_loss', inpaint_loss.item(), step)
        loss = con_loss + inpaint_loss
        print('con inpaint_loss', inpaint_loss.item())
    return loss, loss_img, img_caug, pseudolabel_saug, cutout_map, out_fast

def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann   

# you need to install this as well, there is no pypi package, go here instead: https://github.com/zsef123/Connected_components_PyTorch
from cc_torch import connected_components_labeling

def fix_mirror_padding_gpu(inp):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    Gpu version

    code is for [H,W] tensor
    """
    
    inp = inp.to('cuda')
    cur_max = inp.max()
    # real gain would be if we knew which area we have to search for duplicates -> if we can get a mask from spatial augmenter for this maybe?
    for i in inp.unique()[1:]:
        inst_map = connected_components_labeling((inp==i).byte())
        inst_map_vals = inst_map.unique()
        if inst_map_vals.shape[0]==2:
            continue
        else:
            cnt = 0
            for n,v in enumerate(inst_map_vals[1:]):
                inp[inst_map==v] += cur_max+cnt
                cnt+=1
    return inp

# Test code for comparing speeds:

# y_train = torch.Tensor(Y_train.astype(np.float32))[...,0].to('cuda')

# for img in y_train[:10]:
#     padded = F.pad(img.unsqueeze(0), (20,20,20,20), 'reflect')
#     res = fix_mirror_padding_gpu(padded[0])

# # 222 ms ± 892 µs per loop (mean ± std. dev. of 3 runs, 1 loop each)
    
# y_train = torch.Tensor(Y_train.astype(np.float32))[...,0]

# for img in y_train[:10]:

#     padded = F.pad(img.unsqueeze(0), (20,20,20,20), 'reflect')
#     res = fix_mirror_padding(padded[0].numpy().astype(np.int32))

# # 541 ms ± 111 µs per loop (mean ± std. dev. of 3 runs, 1 loop each)

def make_cpvs(gt_inst, device, background=0):
    # only works for batchsize = 1 and 2d
    gt_inst = gt_inst.squeeze() # H x W or H x W x D
    cpvs = torch.zeros((len(gt_inst.shape),) + gt_inst.shape,
                    dtype=torch.long).to(device)
    labels = gt_inst.unique()
    for label in labels:
        if label == background:
            continue
        x,y = (gt_inst == label).long().nonzero(as_tuple=True)
        cpvs[0,x,y] = (x - x.median()) * -1
        cpvs[1,x,y] = (y - y.median()) * -1
    return cpvs.unsqueeze(0)