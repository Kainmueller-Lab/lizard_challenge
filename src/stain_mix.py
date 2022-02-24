# From https://github.com/aetherAI/stain-mixup
#
# Chang, J.-R., Wu, M.-S., Yu, W.-H., Chen, C.-C., Yang, C.-K., Lin, Y.-Y., & Yeh, C.-Y. (2021). 
# Stain mix-up: Unsupervised domain generalization for histopathology images. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, 117–126. 
# https://doi.org/10.1007/978-3-030-87199-4_11
#
import cv2
import numpy as np

# this needs spams, spams in turn needs  sudo apt-get -y install libblas-dev liblapack-dev gfortran
import spams
import torch

def get_foreground(image: np.ndarray, luminance_threshold: float = 0.8):
    """Get tissue area (foreground)
    Args:
        image: Image in RGB (H, W, C)
        luminance_threshold: cutoff for L
    Return:
        Tissue foreground mask (H, W)
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = image_lab[:, :, 0] / 255.
    return L < luminance_threshold


def rgb_to_od(image):
    od = -np.log(np.maximum(image.astype(np.float32), 1.0) / 255)
    return np.maximum(od, 0)


def od_to_rgb(od):
    image = np.minimum(np.exp(-od) * 255, 255).astype(np.uint8)
    return image


def get_stain_matrix(
    image: np.ndarray,
    lambda1: float = 0.1
):
    """
    Stain matrix estimation via method of:
    A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
    Args:
        image: Image in RGB
        lambda1: lambda1 parameter
    Return:
        stain_matrix
    """
    tissue_mask = get_foreground(image).reshape((-1, ))

    # Convert image to OD (and remove background)
    optical_density = rgb_to_od(image).reshape((-1, 3))
    optical_density = optical_density[tissue_mask]

    stain_matrix = spams.trainDL(
        X=optical_density.T,
        K=2,
        lambda1=lambda1,
        mode=2,
        modeD=0,
        posAlpha=True,
        posD=True,
        verbose=False,
        batchsize=1024,
    ).T  # (N, 3)

    # Assure H on first row, E on second row
    # (NOTE) Only consider H&E in this case.
    # If the numbers of stain is not 2 (ex: H&E), re-alignment is needed.
    if len(stain_matrix) == 2 and stain_matrix[0, 0] < stain_matrix[1, 0]:
        stain_matrix = stain_matrix[[1, 0], :]

    stain_matrix /= np.linalg.norm(stain_matrix, axis=1)[:, None]
    return stain_matrix

def get_stain_matrix_multi(images: np.ndarray, lambda1: float=0.1):
    '''
    same function as above but with entire dataset as input
    '''
    lambda1 = 0.1
    
    stain_grid = []
    for image in images:
        tissue_mask = get_foreground(image).reshape((-1, ))

        # Convert image to OD (and remove background)
        optical_density = rgb_to_od(image).reshape((-1, 3))
        stain_grid.append(optical_density[tissue_mask].T)


    stain_matrix = spams.trainDL(
        X=np.concatenate(stain_grid,axis=-1),
        K=2,
        lambda1=lambda1,
        mode=2,
        modeD=0,
        posAlpha=True,
        posD=True,
        verbose=False,
        batchsize=1024,
    ).T  # (N, 3)

    # Assure H on first row, E on second row
    # (NOTE) Only consider H&E in this case.
    # If the numbers of stain is not 2 (ex: H&E), re-alignment is needed.
    if len(stain_matrix) == 2 and stain_matrix[0, 0] < stain_matrix[1, 0]:
        stain_matrix = stain_matrix[[1, 0], :]

    stain_matrix /= np.linalg.norm(stain_matrix, axis=1)[:, None]
    return stain_matrix


def get_concentration(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    lambda1: float = 0.01,
):
    optical_density = rgb_to_od(image).reshape((-1, 3))
    concentration = spams.lasso(
        X=np.asfortranarray(optical_density.T),
        D=np.asfortranarray(stain_matrix.T),
        mode=2,
        lambda1=lambda1,
        pos=True,
        numThreads=1,
    ).toarray()
    concentration = concentration.T
    concentration = concentration.reshape(*image.shape[:-1], -1)
    return concentration

def stain_mixup(
    image: np.ndarray,
    source_stain_matrix: np.ndarray,
    target_stain_matrix: np.ndarray,
    intensity_range: list = [0.95, 1.05],
    alpha: float = 0.6
) -> np.ndarray:
    """Stain Mix-Up
    Args:
        image: Image array in RGB (H, W, 3)
        source_stain_matrix: Stain matrix of source domain (n_stains, 3); n_stains = 2 for H&E
        target_stain_matrix: Stain matrix of target domain (n_stains, 3); n_stains = 2 for H&E
        intensity_range: The lower bound and upper bound of concentration flucutation
        alpha: The weight of soruce_stain_matrix
    Return:
        Augmented image (H, W, 3)
    """
    n_stains, _ = source_stain_matrix.shape
    # Intensity pertubation
    random_intensity = np.random.uniform(size=(1, 1, n_stains), *intensity_range)  # (1, 1, n_stains)
    src_concentration = get_concentration(
        image,
        source_stain_matrix,
    )  # (H, W, n_stains)
    augmented_concentration = src_concentration * random_intensity

    # Stain matrix intepolation
    interpolated_stain_matrix = source_stain_matrix * alpha + target_stain_matrix * (1. - alpha)
    interpolated_stain_matrix /= np.linalg.norm(
        interpolated_stain_matrix,
        axis=-1,
        keepdims=True,
    )  # (n_stains, 3)

    # Composite
    augmented_image = od_to_rgb(augmented_concentration @ interpolated_stain_matrix)
    return augmented_image

def torch_stain_mixup(raw, source_stain, target_stain, intensity_range=[0.95,1.05], alpha=.6):
    #expecting BCHW
    device = raw.device
    raw = raw.permute(0,2,3,1).cpu().numpy()
    out = []
    for im in raw:
        out.append(stain_mixup(im, source_stain, target_stain, intensity_range, alpha))
    return torch.Tensor(np.stack(out)).to(device)



# stain_train = get_stain_matrix_multi(X_train)
# stain_val = get_stain_matrix_multi(X_val)