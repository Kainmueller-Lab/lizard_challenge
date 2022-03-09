
import itk
import numpy as np
import torch
from torch.utils.data import DataLoader
from .spatial_augmenter import SpatialAugmenter
from .data_utils import SliceDataset, make_pseudolabel, make_instance_segmentation_cl, make_ct,make_reg, instance_wise_connected_components, remove_objects_class,remove_holes
from tqdm.auto import tqdm
from .utils import print_dir, recur_find_ext, save_as_json
from .multi_head_unet import *
import segmentation_models_pytorch as smp

from skimage.morphology import remove_small_objects

def run(
        input_dir: str,
        output_dir: str,
        user_data_dir: str,
    ) -> None:
    """Entry function for automatic evaluation.

    This is the function which will be called by the organizer
    docker template to trigger evaluation run. All the data
    to be evaluated will be provided in "input_dir" while
    all the results that will be measured must be saved
    under "output_dir". Participant auxiliary data is provided
    under  "user_data_dir".

    input_dir (str): Path to the directory which contains input data.
    output_dir (str): Path to the directory which will contain output data.
    user_data_dir (str): Path to the directory which contains user data. This
        data include model weights, normalization matrix etc. .

    """
    # ! DO NOT MODIFY IF YOU ARE UNCLEAR ABOUT API !
    # <<<<<<<<<<<<<<<<<<<<<<<<< 
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    # recursively print out all subdirs and their contents
    print_dir(input_dir)
    print("USER_DATA_DIR: ")
    # recursively print out all subdirs and their contents
    print_dir(user_data_dir)
    print(f"OUTPUT_DIR: {output_dir}")

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH))
    np.save("images.npy", images)
    # >>>>>>>>>>>>>>>>>>>>>>>>>
    params = {'checkpoint_path': 'best_model_b7v1',
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = SliceDataset(raw=images, labels=None)

    encoder = smp.encoders.get_encoder(
            name= "timm-efficientnet-b7",
            in_channels=3,
            depth=5,
            weights=None).to(device)
    decoder_channels = (256, 128, 64, 32, 16)
    decoder_inst = UnetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None).to(device)
    decoder_ct = UnetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None).to(device)
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
    state = torch.load(f'{user_data_dir}/{params["checkpoint_path"]}')
    model.load_state_dict(state['model_state_dict'])
    print(state['step'])
        


    model = model.to(device).train() # MC Dropout
    dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)
    aug_params = {
        'mirror': {'prob_x': 0.5, 'prob_y': 0.5, 'prob': 0.85},
        'translate': {'max_percent':0.05, 'prob': 0.0},
        'scale': {'min': 0.8, 'max':1.2, 'prob': 0.0},
        'zoom': {'min': 0.8, 'max':1.2, 'prob': 0.0},
        'rotate': {'rot90': True, 'prob': 0.85},
        'shear': {'max_percent': 0.1, 'prob': 0.0},
        'elastic': {'alpha': [120,120], 'sigma': 8, 'prob': 0.0}
    }
    augmenter = SpatialAugmenter(aug_params).to(device)
    pred_emb_list = []
    pred_class_list = []
    for raw, _ in tqdm(dataloader):
        raw = raw.to(device).float()
        raw = raw + raw.min() *-1
        raw /= raw.max()
        raw = raw.permute(0,3,1,2) # BHWC -> BCHW        
        with torch.no_grad():
            ct, inst = make_pseudolabel(raw, model, 10, augmenter)
            pred_emb_list.append(inst.squeeze().cpu().detach().numpy())
            pred_class_list.append(ct.cpu().detach())
    


    best_fg_thresh_cl = [0.59, 0.62, 0.70, 0.43, 0.74, 0.57] 
    best_seed_thresh_cl = [0.45, 0.43, 0.80, 0.45, 0.79, 0.72]
    best_min_threshs = [30, 20, 30, 18, 25, 18]
    best_max_threshs = [650, 3000, 600, 400, 600, 1200]
    best_solidity_thresh = [0.5, 0.5, 0.84, 0.79, 0.7, 0.63]

    pred_list = []
    pred_regression = {
        "neutrophil"            : [],
        "epithelial-cell"       : [],
        "lymphocyte"            : [],
        "plasma-cell"           : [],
        "eosinophil"            : [],
        "connective-tissue-cell": [],
    }
    for pred_3c, pred_class in tqdm(zip(pred_emb_list, pred_class_list)):
        pred_inst, _ = make_instance_segmentation_cl(pred_3c, np.argmax(np.squeeze(pred_class.numpy())[1:],axis=0), fg_thresh_cl=best_fg_thresh_cl, seed_thresh_cl=best_seed_thresh_cl)
        pred_inst = remove_holes(pred_inst, max_hole_size=50)
        pred_inst = instance_wise_connected_components(pred_inst)
        pred_inst = torch.tensor(pred_inst.astype(np.int32)).long()
        pred_ct = make_ct(pred_class, pred_inst)
        pred_inst, pred_ct = remove_objects_class(pred_inst, pred_ct, best_min_threshs, best_max_threshs, best_solidity_thresh)
        pred_reg = make_reg(pred_ct, pred_inst)
        for key in pred_regression.keys():
            pred_regression[key].append(pred_reg[key])
        pred_list.append(torch.stack([pred_inst, pred_ct], dim=-1).cpu().detach().numpy())

    for key in pred_regression.keys():
        pred_regression[key] = np.array(pred_regression[key])

    # Valid predictions
    pred_segmentation = np.stack(pred_list, axis=0).astype(np.int32)
    for key in pred_regression.keys():
        pred_regression[key] = np.array(pred_regression[key])

    # ! DO NOT MODIFY IF YOU ARE UNCLEAR ABOUT API !
    # <<<<<<<<<<<<<<<<<<<<<<<<<

    # For segmentation, the result must be saved at
    #     - /output/<uid>.mha
    # with <uid> is can anything. However, there must be
    # only one .mha under /output.
    itk.imwrite(
        itk.image_from_array(pred_segmentation),
        f"{output_dir}/pred_seg.mha"
    )

    # For regression, the result for counting "neutrophil",
    # "epithelial", "lymphocyte", "plasma", "eosinophil",
    # "connective" must be respectively saved at
    #     - /output/neutrophil-count.json
    #     - /output/epithelial-cell-count.json
    #     - /output/lymphocyte-count.json
    #     - /output/plasma-cell-count.json
    #     - /output/eosinophil-count.json
    #     - /output/connective-tissue-cell-count.json
    TYPE_NAMES = [
        "neutrophil",
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "eosinophil",
        "connective-tissue-cell"
    ]
    for _, type_name in enumerate(TYPE_NAMES):
        cell_counts = pred_regression[type_name].astype(np.int32).tolist()
        save_as_json(
            cell_counts,
            f'{output_dir}/{type_name}-count.json'
        )
    # >>>>>>>>>>>>>>>>>>>>>>>>>
