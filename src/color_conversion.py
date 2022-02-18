import torch
import numpy as np
from torchvision.io import read_image, write_jpeg
from torchvision.io.image import ImageReadMode
from torchvision.transforms.transforms import ColorJitter, ConvertImageDtype


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

# log contrast:
# def linear_contrast(img, alpha, e):
#     return alpha*torch.log2(e*2+img)

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



if __name__ == '__main__':
    torch.manual_seed(5)
    img = read_image("/mnt/c/Users/Elias/Desktop/stroma/tiles/993b_B2017.17261_F_HE/993b_B2017.17261_F_HE.mrxs_lvl0_x7331_y46324.jpeg",mode=ImageReadMode.RGB)
    # img = torch.tensor([[[1.]],[[0.]],[[0.]]])
    img = ConvertImageDtype(torch.float)(img)
    
    # for i in range(5):
    img_hed = Rgb2Hed()(img)
    # img_hed = LinearContrast((.25,4.), per_channel=True)(img_hed)
    img_ = Hed2Rgb()(img_hed)
    img_ = ConvertImageDtype(torch.uint8)(img_)
    write_jpeg(img_, f"/mnt/c/Users/Elias/Desktop/hed_augment_test.jpg", quality=100)

    print("test")
