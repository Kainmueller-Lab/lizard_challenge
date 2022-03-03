import torch
# need this for it to work, maybe we just skip this for now.
from cc_torch import connected_components_labeling

# def fix_mirror_padding(ann):
#     """Deal with duplicated instances due to mirroring in interpolation
#     during shape augmentation (scale, rotation etc.).
#     """
#     current_max_id = np.amax(ann)
#     inst_list = list(np.unique(ann))
#     inst_list.remove(0)  # 0 is background
#     for inst_id in inst_list:
#         inst_map = np.array(ann == inst_id, np.uint8)
#         remapped_ids = measurements.label(inst_map)[0]
#         remapped_ids[remapped_ids > 1] += current_max_id
#         ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
#         current_max_id = np.amax(ann)
#     return ann

def fix_mirror_padding_gpu(inp):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    Gpu version

    code is for [H,W] tensor
    """
    
    inp = inp.to('cuda') # remove if this is done before already
    cur_max = inp.max()
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
    
    
# speed example for main.py
# y_train = torch.Tensor(Y_train.astype(np.float32))[...,0].to('cuda')
# for img in y_train:
#     print(img.shape)
#     padded = F.pad(img.unsqueeze(0), (20,20,20,20), 'reflect')
#     res = fix_mirror_padding_gpu(padded[0])
    
    

