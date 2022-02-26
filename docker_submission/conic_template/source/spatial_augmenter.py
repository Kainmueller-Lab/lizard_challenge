import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.transforms import GaussianBlur

class SpatialAugmenter(torch.nn.Module,):

    def __init__(self, params, interpolation='bilinear', padding_mode = 'zeros'):
        '''
        params= {
            'mirror': {'prob': float [0,1], 'prob_x': float [0,1],'prob_y': float [0,1]},
            'translate': {'max_percent':float [0,1], 'prob': float [0,1]},
            'scale': {'min': float, 'max':float, 'prob': float [0,1]},
            'zoom': {'min': float, 'max':float, 'prob': float [0,1]},
            'rotate': {'rot90': bool, 'max_degree': int [0,360], 'prob': float [0,1]},
            'shear': {'max_percent': float [0,1], 'prob': float [0,1]},
            'elastic': {'alpha': list[float|int], 'sigma': float|int, 'prob': float [0,1]}}
        '''
        super(SpatialAugmenter, self).__init__()
        self.params = params
        self.mode = 'forward'
        self.random_state = {}
        # fill dict so that augmentation functions can be tested
        for key in self.params.keys():
            self.random_state[key] = {}
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        
    def forward_transform(self, img, label=None, random_state=None):
        self.mode = 'forward'
        self.device = img.device
        if random_state:
            self.random_state = random_state
        else:
            for key in self.params.keys():
                self.random_state[key] = {'prob' : bool(np.random.binomial(1,self.params[key]['prob']))}
        for key in list(self.params.keys()):
            if self.random_state[key]['prob']:
                #print('Do transform: ', key)
                func = getattr(self, key)
                img, label = func(img, label=label, random_state=random_state)
        if label is not None:
            return img, label
        else:
            return img
        
    def inverse_transform(self, img, label=None, random_state=None):
        self.mode = 'inverse'
        self.device = img.device
        keylist = list(self.params.keys())
        keylist.reverse()
        if random_state:
            self.random_state = random_state
        for key in keylist:
            if self.random_state[key]['prob']:
                #print('Do inverse transform: ', key)
                func = getattr(self, key)
                img, label = func(img, label=label)        
        if label is not None:
            return img, label
        else:
            return img
            
    def mirror(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            self.random_state['mirror']['x'] = bool(np.random.binomial(1,self.params['mirror']['prob_x']))
            self.random_state['mirror']['y'] = bool(np.random.binomial(1,self.params['mirror']['prob_y']))
            #
        x = self.random_state['mirror']['x']
        y = self.random_state['mirror']['y']
        if x:
            x = -1
        else:
            x = 1
        if y:
            y = -1
        else:
            y = 1
        theta = torch.tensor([[[x  , 0.0, 0.0],
                               [0.0, y  , 0.0]]])
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None

    def translate(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            x = np.random.uniform(-self.params['translate']['max_percent'],self.params['translate']['max_percent'])
            y = np.random.uniform(-self.params['translate']['max_percent'],self.params['translate']['max_percent'])
            self.random_state['translate']['x'] = x
            self.random_state['translate']['y'] = y
        elif self.mode == 'inverse':
            x = -1 * self.random_state['translate']['x']
            y = -1 * self.random_state['translate']['y']
        else:
            x = self.random_state['translate']['x']
            y = self.random_state['translate']['y']
        theta =  torch.tensor([[[1.0, 0.0, x],
                                [0.0, 1.0, y]]])
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None

    def zoom(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            zoom_factor = np.random.uniform(self.params['scale']['min'],self.params['scale']['max'])
            self.random_state['zoom']['factor'] = zoom_factor
        elif self.mode == 'inverse':
            zoom_factor = 1/self.random_state['zoom']['factor']
        else:
            zoom_factor = self.random_state['zoom']['factor']
        theta = torch.tensor([[[zoom_factor  , 0.0, 0.0],
                               [0.0, zoom_factor  , 0.0]]])
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None
        
    def scale(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            x = np.random.uniform(self.params['scale']['min'],self.params['scale']['max'])
            y = np.random.uniform(self.params['scale']['min'],self.params['scale']['max'])
            self.random_state['scale']['x'] = x
            self.random_state['scale']['y'] = y
        elif self.mode == 'inverse':
            x = 1/self.random_state['scale']['x']
            y = 1/self.random_state['scale']['y']
        else:
            x = self.random_state['scale']['x']
            y = self.random_state['scale']['y']
        theta = torch.tensor([[[x  , 0.0, 0.0],
                               [0.0, y  , 0.0]]])
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None

    def rotate(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            if 'rot90' in self.params['rotate'].keys() and self.params['rotate']['rot90']:
                degree = np.random.choice([-270,-180,-90,90,180,270])
            else:
                degree = np.random.uniform(-self.params['rotate']['max_degree'], self.params['rotate']['max_degree'])
            self.random_state['rotate']['degree'] = degree
        elif self.mode == 'inverse':
            degree = -1 * self.random_state['rotate']['degree']
        else:
            degree = self.random_state['rotate']['degree']
        rad = np.deg2rad(degree)
        theta = torch.tensor([[[np.cos(rad),  -np.sin(rad), 0.0],
                               [np.sin(rad),   np.cos(rad), 0.0]]]).float()
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None

    def shear(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            x = np.random.uniform(-self.params['shear']['max_percent'], self.params['shear']['max_percent'])
            y = np.random.uniform(-self.params['shear']['max_percent'], self.params['shear']['max_percent'])
            self.random_state['shear']['x'] = x
            self.random_state['shear']['y'] = y
        elif self.mode == 'inverse':
            x = -self.random_state['shear']['x']
            y = -self.random_state['shear']['y']
        else:
            x = self.random_state['shear']['x']
            y = self.random_state['shear']['y']            
        theta = torch.tensor([[[1.0, x  , 0.0], 
                               [y  , 1.0, 0.0]]])
        grid = F.affine_grid(theta, img.size()).to(self.device)
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None
                 
    def identity_grid(self, img):
        theta = torch.tensor([[[1.0  , 0.0, 0.0],
                               [0.0, 1.0 , 0.0]]])
        return F.affine_grid(theta, img.size()).to(self.device)
    
    def elastic(self, img, label, random_state=None):
        if self.mode == 'forward' and not random_state:
            displacement = self.create_elastic_transformation(shape = list(img.shape[-2:]),
                                                         alpha=self.params['elastic']['alpha'],
                                                         sigma = self.params['elastic']['sigma'])
            self.random_state['elastic']['displacement'] = displacement
        elif self.mode == 'inverse':
            displacement = -1 * self.random_state['elastic']['displacement']
        else:
            displacement = self.random_state['elastic']['displacement']
        identity_grid = self.identity_grid(img)
        grid =  identity_grid + displacement
        if label is not None:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), F.grid_sample(label, grid, mode='nearest', padding_mode=self.padding_mode)
        else:
            return F.grid_sample(img, grid, mode=self.interpolation, padding_mode=self.padding_mode), None
    
    def create_elastic_transformation(self, shape, alpha = [80,80], sigma = 8):

        blur = GaussianBlur(kernel_size=int(8*sigma+1), sigma=sigma)
        dx = blur(torch.rand(*shape).unsqueeze(0).unsqueeze(0)*2 -1).to(self.device) * alpha[0]/shape[0]
        dy = blur(torch.rand(*shape).unsqueeze(0).unsqueeze(0)*2 -1).to(self.device) * alpha[1]/shape[1]

        displacement = torch.concat([dx,dy],1).permute([0,2,3,1])
        return displacement