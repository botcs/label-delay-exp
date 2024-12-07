import time
from typing import Any, Dict, List

import omegaconf
import torch
import torch.nn as nn

from solo.methods.base import BaseMethod


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class AdaBN(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.before_tta_checkpoint = None

    def configure_adabn_model(self):
        """Configure model for use with AdaBN."""
        # train mode, because tent optimizes the model to minimize entropy
        self.backbone.train()
        # disable grad
        self.backbone.requires_grad_(False)
        self.classifier.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                ######!!!!!! NOTICE here. For TTA methods we always use the current batch for mean and var for the BN layers
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None


    def copy_model(self):
        # just keep a copy in memory for branching out parameters for TTA
        backbone_state = self.backbone.state_dict()
        classifier_state = self.classifier.state_dict()
        optimizers_state = self.optimizers().state_dict()
        return backbone_state, classifier_state, optimizers_state
        

    def load_model(self, backbone_state, classifier_state, optm_state):
        # load the model after evaluation

        # load the running stats of batch norm specifically
        # (the load_state_dict() function does not work for this)
        for nm, m in self.backbone.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.track_running_stats = True
                m.running_mean = backbone_state[nm + ".running_mean"]
                m.running_var = backbone_state[nm + ".running_var"]
                
        
        # load the rest of the model
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.classifier.load_state_dict(classifier_state, strict=True)
        self.optimizers().load_state_dict(optm_state)


    def unsupervised_step(self, unsupervised_data, batch_idx, logged_metrics, outs):
        self.configure_adabn_model()
        # just take a forward pass, no need to compute gradients
        super().unsupervised_step(unsupervised_data, batch_idx, logged_metrics, outs)
        
        
    def training_step(self, batch: List[Any], batch_idx: int):
        
        eval_data, unsupervised_data, supervised_data = batch
        outs = {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
        logged_metrics = {}

        iter_start = time.time()
        # Online Eval
        self.eval_step(eval_data, batch_idx, logged_metrics, outs)

        # Restore model to train mode
        if batch_idx % self.cfg.online.batch_repeat == 0:
            if self.before_tta_checkpoint is not None:
                self.load_model(*self.before_tta_checkpoint)
            self.unfreeze()
            self.train()

        # Supervised training
        self.supervised_step(supervised_data, batch_idx, logged_metrics, outs)
        
        # test time adaptation
        if batch_idx % self.cfg.online.batch_repeat == self.cfg.online.batch_repeat - 1:
            self.before_tta_checkpoint = self.copy_model()
            self.unsupervised_step(unsupervised_data, batch_idx, logged_metrics, outs)

        self.iter_time.update(time.time() - iter_start)

        # Logging
        if batch_idx < 100:
            # reset timers
            self.eval_time.reset()
            self.unsup_time.reset()
            self.sup_time.reset()
            self.iter_time.reset()
        else:
            logged_metrics.update({
                "eval_time": self.eval_time.compute(),
                "unsup_time": self.unsup_time.compute(),
                "sup_time": self.sup_time.compute(),
                "iter_time": self.iter_time.compute(),
            })

        self.log_dict(logged_metrics, on_epoch=True, sync_dist=True)
        return outs
