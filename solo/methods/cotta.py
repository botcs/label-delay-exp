from copy import deepcopy
from typing import Any, Dict

# import pytorch_lightning as pl
import omegaconf
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda

from solo.methods.tent import TENT


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor
def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5
    
    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0), 
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms


class COTTA(TENT):
    """
    Notes for COTTA:
    optimization step happens inside of the TTA, so self._manual_backward_and_update() only takes care of the scheduler
    Hyperparameter for COTTA:
    cfg.method_kwargs.warmup
    cfg.method_kwargs.augment

    When tuning the hyperparameter, tunning scale_loss and tta_lr is equalevent

    
    """

    
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.backbone, self.tta_optimizer())
        self.transform = get_tta_transforms()    
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                std = [ 1., 1., 1. ])
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.augment = cfg.method_kwargs.augment
    def tta_optimizer(self):
        # return torch.optim.Adam(self.tta_parameters,
        #                         lr=self.cfg.method_kwargs.tent_lr,
        #                         betas=(0.9, 0.999),
        #                         weight_decay=0.0)
        parameter = self.tta_parameters()

        return torch.optim.SGD(parameter,
                   lr=self.cfg.method_kwargs.tta_lr,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=0.0,
                   nesterov=True)

    def extra_model_forward(self,model,X):

        

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = model(X)
 
        logits = self.classifier(feats)

        return logits


    @torch.enable_grad() 
    def tta_procedure(self,outs: dict, X,  **kwargs):
        outputs = outs["logits"][0]
        opt_tta,_ = self.optimizers()
        x = X[0].flatten(0, 1)
        self.model_ema.train()
        # Teacher Prediction
        
        anchor_out = self.extra_model_forward(self.model_anchor,  x)
        anchor_prob = torch.nn.functional.softmax(anchor_out, dim=1).max(1)[0]
        
        standard_ema = self.extra_model_forward(self.model_ema,  x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        print(anchor_prob.mean(0))
        to_aug = anchor_prob.mean(0)<0.1
        self.log('aug',anchor_prob.mean(0),sync_dist=True)
        if to_aug: 
            for i in range(N):
                x=self.denormalize(x)
                outputs_  = self.extra_model_forward(self.model_ema,self.normalize(self.transform(x)) ).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) 
        outs['unsupervised_loss'] = loss
        opt_tta.zero_grad()
        loss.backward()
        opt_tta.step()
        
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.backbone, alpha_teacher=0.999)
        # Stochastic restore
        if True:
            for nm, m  in self.backbone.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.001).float().cuda() 
                        with torch.no_grad():
                            state = self.model_state[f"{nm}.{npp}"].detach()
                            state = state.cuda()
                            p.data = state * mask + p * (1.-mask)
        return outputs_ema
    def _manual_backward_and_update(self, outputs: Dict[str, Any], batch_idx: int):
        """
        Call the manual optimization step depending on which iteration are we at.


        For the sake of clarity we split the procedure into the following steps:

        1. Evaluation (E): take the evaluation batch and compute the online accuracy
                with the most recent backbone + classifier parameters
        2. Unsupervised (U): take the unsupervised batch and compute the unsupervised loss
                with the checkpoint of the backbone from the last (U) update
                followed by that, create a new checkpoint of the backbone
        3. Supervised (S): take the supervised batch and compute the supervised loss
                with the checkpoint of the backbone from the last (U) checkpoint
        

        Since it is not trivial how do we do alternating optimization, we generalize the steps:
            we take a string of letters "E" / "U" / "S" and we perform the corresponding step.

        Note that the following rules apply:
            - Each timestep is started by an E
            - There is one and only one E in each timestep
            - There can be multiple U and S in each timestep
            - The U steps always preceed the S steps
        """
        assert self._updated == False

        timestep = batch_idx // self.num_steps_per_timestep
        sequence_idx = batch_idx % self.num_steps_per_timestep
        step_type = self.step_sequence[sequence_idx]

        unsup_opt, sup_opt = self.optimizers()
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            unsup_sch, sup_sch = schedulers
        else:
            unsup_sch, sup_sch = None, None


        if step_type == "E":
            assert "next_batch_acc1" in outputs, "Next batch accuracy not found in outputs, while it's EVAL step"
            assert "unsupervised_loss" not in outputs, "Unsupervised loss found in outputs, while it's not UNSUPERVISED step"
            assert "supervised_loss" not in outputs, "Supervised loss found in outputs, while it's not SUPERVISED step"

        elif step_type == "U":
            assert "next_batch_acc1" not in outputs, "Next batch accuracy found in outputs, while it's not EVAL step"
            assert "unsupervised_loss" in outputs, "Unsupervised loss not found in outputs, while it's UNSUPERVISED step"
            assert "supervised_loss" not in outputs, "Supervised loss found in outputs, while it's not SUPERVISED step"


            if batch_idx // self.num_steps_per_timestep > self.tta_warmup :
                unsup_sch.step() if unsup_sch is not None else None


        elif step_type == "S" and timestep >= self.online_delay:
            assert "next_batch_acc1" not in outputs, "Next batch accuracy found in outputs, while it's not EVAL step"
            assert "unsupervised_loss" not in outputs, "Unsupervised loss found in outputs, while it's not UNSUPERVISED step"
            assert "supervised_loss" in outputs, "Supervised loss not found in outputs, while it's SUPERVISED step"

            sup_opt.zero_grad()
            self.manual_backward(outputs["supervised_loss"])
            sup_opt.step()
            sup_sch.step() if sup_sch is not None else None
        
        self._updated = True
    
def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)