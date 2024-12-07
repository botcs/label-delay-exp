import math
import time
from copy import deepcopy
from typing import Any, List
import logging
import time
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import copy
import numpy as np
import omegaconf
import torch
import torch.nn as nn

from solo.methods.tent import TENT, softmax_entropy


class SAR(TENT):
    """
    Notes for SAR:
    optimization step happens inside of the TTA, so self._manual_backward_and_update() only takes care of the scheduler
    Hyperparameter for SAR:
    cfg.method_kwargs.scale_loss1: scale of the TTA loss in the first gradient step
    cfg.method_kwargs.scale_loss2: scale of the TTA loss in the second gradient step
    cfg.method_kwargs.scale_margin: scale for the sample selection
    cfg.method_kwargs.recovery: score for the recovery
    self.cfg.method_kwargs.tta_lr: learning rate for TTA parameters. 

    When tuning the hyperparameter, tunning scale_loss and tta_lr is equalevent

    
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super(TENT, self).__init__(cfg)
        self.ema = None
        self.scale1 = cfg.method_kwargs.scale_loss1
        self.scale2 = cfg.method_kwargs.scale_loss2
        self.margin = cfg.method_kwargs.scale_margin
        self.recovery = cfg.method_kwargs.recovery
        self.tta_warmup = cfg.method_kwargs.warmup
        self.step_sequence = "E" +  "S" * self.num_supervised + "U" * self.num_unsupervised 
        self.num_steps_per_timestep = len(self.step_sequence)
        self.configure_tta_model()

    def tta_parameters(self):
        """Collect the affine scale + shift parameters from norm layers.
            Walk the model's modules and collect all normalization parameters.
            Return the parameters and their names.
            Note: other choices of parameterization are possible!
            """
        params = []
        for nm, m in self.backbone.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(
                            {'name': f"{nm}.{np}",
                             "params": p,
                             'lr': self.cfg.method_kwargs.tta_lr
                             }
                        )

        return params





    def tta_optimizer(self):
        base_optimizer = torch.optim.SGD
        return SAM(self.tta_parameters(), base_optimizer, lr=self.cfg.method_kwargs.tta_lr, momentum=0.9)
    
    def recover_model(self, optimizer, optm_state):
        
        self.copy_sup_backbone_to_tta_backbone()
        optimizer.load_state_dict(optm_state)

        

    def tta_procedure(self,outs: dict, X, **kwargs):
        margin = self.margin * math.log(712)
        self.recovery_flag = True
        opt_tta,_ = self.optimizers()
        opt_tta.zero_grad()
        optimizer_state = deepcopy(opt_tta.state_dict())


        # SAR tta, perform 2 gradient steps
        entropys = softmax_entropy(outs["logits"][0])
        mask1 = torch.zeros_like(entropys)
        mask1[entropys < margin] = 1
        outs['unsupervised_loss'] = 0.0
        if mask1.sum() > 0:
            entropys = entropys[mask1.bool()]
            loss = self.scale1 * entropys.mean()
            self.log('first step loss',loss,sync_dist=True)
            self.manual_backward(loss)
            opt_tta.first_step(zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            unsupervised_outs = [self.tta_forward(x.flatten(0, 1)) for x in X[: self.num_large_crops]]
            for k in unsupervised_outs[0].keys():
                outs[k] = [out[k] for out in unsupervised_outs]
            entropys2 = softmax_entropy(outs["logits"][0])
            mask2 = torch.zeros_like(entropys2)
            mask2[entropys2 < margin] = 1  
            # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
            if mask2.sum() > 0:

                entropys2 = entropys2[ (mask1 * mask2).bool() ]  # second time forward
                loss_second = self.scale2 * entropys2.mean()
                if not np.isnan(loss_second.item()):
                    self.ema = update_ema(self.ema,loss_second.item())  # record moving average loss values for model recovery
                    outs['unsupervised_loss'] = loss_second
                # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                self.log('second step loss',loss_second,sync_dist=True)
                self.manual_backward(loss_second)
                opt_tta.second_step(zero_grad=True)
                # perform model recovery
                self.log('recover score', self.ema, sync_dist=True)
                if self.ema is not None:
                    if self.ema > self.recovery:
                        self.recovery_flag = False
        if self.recovery_flag:
            self.recover_model(opt_tta,optimizer_state )
            outs['unsupervised_loss'] = 0.0
                            
                    


        opt_tta.zero_grad()


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


            if batch_idx // self.num_steps_per_timestep > self.tta_warmup and not self.recovery:
                unsup_sch.step() if unsup_sch is not None else None


        elif step_type == "S"and timestep >= self.online_delay:
            assert "next_batch_acc1" not in outputs, "Next batch accuracy found in outputs, while it's not EVAL step"
            assert "unsupervised_loss" not in outputs, "Unsupervised loss found in outputs, while it's not UNSUPERVISED step"
            assert "supervised_loss" in outputs, "Supervised loss not found in outputs, while it's SUPERVISED step"

            sup_opt.zero_grad()
            self.manual_backward(outputs["supervised_loss"])
            sup_opt.step()
            sup_sch.step() if sup_sch is not None else None
        
        self._updated = True





def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
