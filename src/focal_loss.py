import torch
import torch.nn.functional as F

class FocalCE(torch.nn.Module,):
    '''
    Focal Cross Entropy Loss with exponential moving average class weights
    '''
    def __init__(self, num_classes, momentum=0.99, focal_p=3.0, reduction='mean', smoothing=0.05, ema=True):
        super(FocalCE, self).__init__()
        self.num_classes = num_classes
        self.running_conf = torch.ones(num_classes).float()/num_classes
        self.momentum = momentum
        self.reduction = reduction
        self.focal_p = focal_p
        self.smoothing = smoothing
        self.ema = ema

    def _update_running_conf(self, probs, tolerance=1e-8):
        """Maintain the moving class prior"""
        B,C,H,W = probs.size()
        probs_avg = probs.mean(0).view(C,-1).mean(-1)

        if self.ema:
            # use the moving average for the rest
            self.running_conf *= self.momentum
            self.running_conf += (1 - self.momentum) * probs_avg
        else:
            # updating the new records: copy the value
            new_index = probs_avg > tolerance
            self.running_conf[new_index] = probs_avg[new_index]
    
    def _focal_ce(self, logits, target):
        focal_weight = (1 - self.running_conf.clamp(0.)) ** self.focal_p
        return F.cross_entropy(logits, target.squeeze(0), weight=focal_weight, reduction=self.reduction, label_smoothing=self.smoothing)
    
    def forward(self, input, target):
        device = input.device
        if self.running_conf.device != device:
            self.running_conf = self.running_conf.to(device)
        self._update_running_conf(F.one_hot(target.squeeze(0).long(), num_classes=self.num_classes).permute([0,-1,1,2]).float())
        return self._focal_ce(input, target)