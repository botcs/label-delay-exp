# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F

class Pseudo(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.scale_loss: float = cfg.method_kwargs.scale_loss
        self.temperature: float = cfg.method_kwargs.temperature
        self.step_sequence = "E"
        i = self.num_supervised
        j = self.num_unsupervised
        while i > 0 or j > 0:
            if i > 0:
                self.step_sequence += "S"
                i -= 1
            if j > 0:
                self.step_sequence += "U"
                j -= 1

        assert self.num_crops == 1, "Current implementation only works with a single crop"
        initialize_momentum_params(self.classifier, self.momentum_classifier)

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        extra_momentum_pairs = [(self.classifier, self.momentum_classifier)]
        return super().momentum_pairs + extra_momentum_pairs

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(Pseudo, Pseudo).add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.scale_loss = omegaconf_select(cfg, "method_kwargs.scale_loss", 1.0)
        cfg.method_kwargs.temperature = omegaconf_select(cfg, "method_kwargs.temperature", 1.0)
        cfg.momentum.classifier = True


        return cfg


    def _compute_pseudo_label_loss(self, out: Dict[str, Any]) -> torch.Tensor:
        """Computes the pseudo-label loss.

        Args:
            out (Dict[str, Any]): output of the forward method.

        Returns:
            torch.Tensor: pseudo-label loss.
        """


        # Compute soft cross-entropy loss
        teacher_logits = out["momentum_logits"][0].detach()
        student_logits = out["logits"][0]
        
        # Apply temperature scaling
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Compute soft cross-entropy loss
        # This multiplies the log probabilities of the student by the probabilities of the teacher and sums across classes
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss
        

        
    def unsupervised_step(self, batch, batch_idx):
        out = super().unsupervised_step(batch, batch_idx)
        pseudo_label_loss = self._compute_pseudo_label_loss(out)

        unsupervised_loss = pseudo_label_loss * self.scale_loss
        out["unsupervised_loss"] = unsupervised_loss
        self.log("unsupervised_loss", unsupervised_loss, sync_dist=True)

        return out
