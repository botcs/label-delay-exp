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

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import torchvision

import torchmetrics
import omegaconf
# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    resnet18,
    resnet50,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    wide_resnet28w2,
    wide_resnet28w8,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import omegaconf_select, remove_bias_and_norm_from_weight_decay
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from torch.optim.lr_scheduler import MultiStepLR
import time


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)


class BaseMethod(pl.LightningModule):
    _BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
        "wide_resnet28w2": wide_resnet28w2,
        "wide_resnet28w8": wide_resnet28w8,
    }
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(self, cfg: omegaconf.DictConfig):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        Cfg basic structure:
            backbone:
                name (str): architecture of the base backbone.
                kwargs (dict): extra backbone kwargs.
            data:
                dataset (str): name of the dataset.
                num_classes (int): number of classes.
            max_epochs (int): number of training epochs.

            backbone_params (dict): dict containing extra backbone args, namely:
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                classifier_lr (float): learning rate for the online linear classifier.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                    step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.
            knn_eval:
                enabled (bool): enables online knn evaluation while training.
                k (int): the number of neighbors to use for knn.
            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly
            if using gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()
        self.save_hyperparameters()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)

        self.cfg: omegaconf.DictConfig = cfg

        ########## Backbone ##########
        self.backbone_args: Dict[str, Any] = cfg.backbone.kwargs
        assert cfg.backbone.name in BaseMethod._BACKBONES
        self.base_model: Callable = self._BACKBONES[cfg.backbone.name]
        self.backbone_name: str = cfg.backbone.name
        # initialize backbone
        kwargs = self.backbone_args.copy()

        method: str = cfg.method
        self.backbone: nn.Module = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            self.features_dim: int = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
        else:
            self.features_dim: int = self.backbone.num_features
        ##############################


        ########## Supervision params ##########
        self.supervised_backbone = None
        self.supervised_projection_head = None
        self.supervised_features_dim = self.features_dim
        self.supervised_param_names = cfg.online.supervised_param_names
        self.share_backbone = cfg.online.share_backbone
        self.online_delay = cfg.online.delay


        ########## Supervised Backbone ##########
        if self.share_backbone:
            self.supervised_backbone = self.backbone
        else:
            if "backbone" in cfg.online.supervised_param_names:
                kwargs = self.backbone_args.copy()
                self.supervised_backbone: nn.Module = self.base_model(cfg.method, **kwargs)
                if self.backbone_name.startswith("resnet"):
                    # remove fc layer
                    self.supervised_backbone.fc = nn.Identity()


        ########## Supervised Projection Head ##########
        if "projection_head" in cfg.online.supervised_param_names:
            assert cfg.online.supervised_projection_head.enabled
            self.supervised_projection_head = self._build_mlp(
                num_layers=cfg.online.supervised_projection_head.num_layers,
                input_dim=self.features_dim,
                mlp_dim=cfg.online.supervised_projection_head.hidden_dim,
                output_dim=cfg.online.supervised_projection_head.output_dim,
                last_bn=True
            )
            self.supervised_features_dim = cfg.online.supervised_projection_head.output_dim

        ########## Supervised Classifier ##########
        self.num_classes: int = cfg.data.num_classes
        self.classifier: nn.Module = nn.Linear(self.supervised_features_dim, self.num_classes)

        # training related
        self.max_epochs: int = cfg.max_epochs

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.classifier_lr: float = cfg.optimizer.classifier_lr
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )


        # data-related
        self.num_large_crops: int = cfg.data.num_large_crops
        self.num_small_crops: int = cfg.data.num_small_crops
        self.num_crops: int = self.num_large_crops + self.num_small_crops
        # turn on multicrop if there are small crops
        self.multicrop: bool = self.num_small_crops != 0

        # [author1]
        # Delayed labeling requires a different optimizer when used with SSL
        self.automatic_optimization = False
        self.supervised_param_names = cfg.online.supervised_param_names
        self.num_supervised = cfg.online.num_supervised
        self.num_unsupervised = cfg.online.num_unsupervised
        self.step_sequence = "E" + "U" * self.num_unsupervised + "S" * self.num_supervised
        self.num_steps_per_timestep = len(self.step_sequence)
        """
        explanation for num_steps_per_timestep:
        1: Evaluation iteration
        self.num_supervised: Supervised iteration(s)
        self.num_unsupervised: Unsupervised iteration(s)


        1. Evaluation (E): take the evaluation batch and compute the online accuracy
                with the most recent backbone + classifier parameters
        2. Unsupervised (U): take the unsupervised batch and compute the unsupervised loss
                with the checkpoint of the backbone from the last (U) update
                followed by that, create a new checkpoint of the backbone
        3. Supervised (S): take the supervised batch and compute the supervised loss
                with the checkpoint of the backbone from the last (U) checkpoint
        """

        # for online next-batch evaluation
        self.next_batch_acc1 = torchmetrics.MeanMetric()
        self.next_batch_acc1.persistent(True)

        # for performance logging
        self.eval_time = torchmetrics.MeanMetric()
        self.unsup_forward_time = torchmetrics.MeanMetric()
        self.sup_forward_time = torchmetrics.MeanMetric()
        self.backward_and_update_time = torchmetrics.MeanMetric()
        self.data_time = torchmetrics.MeanMetric()
        self.iter_time = torchmetrics.MeanMetric()
        self.time_metrics = {
            "eval_time": self.eval_time,
            "unsup_forward_time": self.unsup_forward_time,
            "sup_forward_time": self.sup_forward_time,
            "backward_and_update_time": self.backward_and_update_time,
            "data_time": self.data_time,
            "iter_time": self.iter_time,
        }

        # make sure that metrics are saved with the checkpoint as part of the state_dict
        for metric in self.time_metrics.values():
            metric.persistent(True)

        self.data_start = time.time()
        self._updated = False

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default for extra backbone kwargs (use pytorch's default if not available)
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})


        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        # default empty parameters for method-specific kwargs
        cfg.method_kwargs = omegaconf_select(cfg, "method_kwargs", {})

        cfg.online.share_backbone = omegaconf_select(cfg, "online.share_backbone", True)

        return cfg

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        return [
            # {"name": "all", "params": self.parameters()},
            {"name": "backbone", "params": self.backbone.parameters()},
        ]

    @property
    def supervised_learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the supervised branch.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        params = [{"name": "classifier", "params": self.classifier.parameters()}]
        if "backbone" in self.supervised_param_names:
            params.append({
                "name": "supervised_backbone", "params": self.supervised_backbone.parameters()
            })

        if "projection_head" in self.supervised_param_names:
            params.append({
                "name": "supervised_projection_head", "params": self.supervised_projection_head.parameters()
            })

        return params

    def configure_optimizers(self) -> Tuple[List, List]:
        learnable_params = self.learnable_params

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)


        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # sup_optimizer = torch.optim.Adam(
        #     self.supervised_learnable_params,
        #     lr=self.classifier_lr,
        #     weight_decay=1e-5,
        # )
        sup_optimizer = torch.optim.SGD(
            self.supervised_learnable_params,
            lr=self.classifier_lr,
            weight_decay=1e-5,
            momentum=0.9
        )


        schedulers = self.configure_schedulers(optimizer, sup_optimizer)
        if schedulers is None:
            return [optimizer, sup_optimizer]
        return [optimizer, sup_optimizer], schedulers

    def configure_schedulers(self, unsup_optimizer, sup_optimizer) -> List:
        scheduler = None

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    unsup_optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.max_epochs, eta_min=self.min_lr
            )
        elif self.scheduler == "none":
            return None
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        # indexes of parameters without lr scheduler
        idxs_no_scheduler = [i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)]

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        # TODO: if necessary write a custom scheduler for the supervised branch
        sup_scheduler = scheduler.copy()

        return [scheduler, sup_scheduler]


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """
        This improves performance marginally. It should be fine
        since we are not affected by any of the downsides descrited in
        https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad

        Implemented as in here
        https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#set-grads-to-none
        """
        try:
            optimizer.zero_grad(set_to_none=True)
        except:
            optimizer.zero_grad()

    '''
    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)

        return {"feats": feats}

    def supervised_forward(self, X) -> Dict:
        # Perform the forward pass
        outs = {}
        if "backbone" in self.supervised_param_names:
            if not self.no_channel_last:
                X = X.to(memory_format=torch.channels_last)

            feats = self.supervised_backbone(X)
            outs["supervised_feats"] = feats
        else:
            with torch.no_grad():
                outs.update(self(X))
                outs["supervised_feats"] = outs["feats"]

        if "projection_head" in self.supervised_param_names:
            outs["supervised_feats"] = self.supervised_projection_head(outs["supervised_feats"])


        logits = self.classifier(outs["supervised_feats"])
        outs["logits"] = logits
        return outs

    '''
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        # Perform the forward pass
        outs = {}
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        logits = self.classifier(feats)
        outs["feats"] = feats
        outs["logits"] = logits
        return outs

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Basic multicrop forward method that performs the forward pass
        for the multicrop views. Children classes can override this method to
        add new outputs but should still call this function. Make sure
        that this method and its overrides always return a dict.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        return {"feats": feats}


    def quick_eval(self):
        if len(self.eval_modules) == 0:
            self.eval()
        else:
            for module in self.eval_modules:
                module.eval()

    def quick_train(self):
        if len(self.eval_modules) == 0:
            self.train()
        else:
            for module in self.eval_modules:
                module.train()

    def log_histogram(self, log_histograms: Dict[str, torch.Tensor]):
        if self.timestep % self.cfg.log_histogram_every_n_steps == 0:
            # Currently you can't log tensor histograms only scalars in the PL api
            if any(isinstance(l, pl.loggers.wandb.WandbLogger) for l in self.loggers):
                wandb = pl.loggers.wandb.wandb
                # check if wandb was initialized
                if wandb.run is not None:
                    log_dict = {
                        f"histogram/{k}": wandb.Histogram(v.tolist())
                        for k, v in log_histograms.items()
                    }
                    log_dict["timestep"] = self.timestep
                    log_dict["global_step"] = self.global_step
                    wandb.log(log_dict)

    def eval_step(self, eval_data, batch_idx: int) -> Dict:
        ##############################
        # EVALUATION ON NEXT BATCH
        ##############################
        self.eval()
        self.freeze()
        # self.quick_eval()
        with torch.no_grad():
            eval_ind, eval_x, eval_target = eval_data

            batch_size = len(eval_x)
            eval_next_x = eval_x
            eval_next_target = eval_target

            eval_next_logit = self.forward(eval_next_x)["logits"]
            eval_next_pred = torch.argmax(eval_next_logit, dim=1)
            # eval_next_pred_entropy = softmax_entropy(eval_next_logit).mean()

            eval_next_top1_softmax_values, _ = torch.topk(
                eval_next_logit.softmax(dim=1), k=1, dim=1
            )
            if self.num_classes > 5:
                eval_next_acc1, eval_next_acc5 = accuracy_at_k(eval_next_logit, eval_next_target, top_k=[1, 5])
            else:
                eval_next_acc1 = accuracy_at_k(eval_next_logit, eval_next_target, top_k=[1])[0]
                eval_next_acc5 = 1.0



            # select those softmax values that have been correctly classified
            correct_mask = eval_next_pred == eval_next_target
            correct_top1_softmax_values = eval_next_top1_softmax_values[correct_mask]
            incorrect_top1_softmax_values = eval_next_top1_softmax_values[~correct_mask]

            out = {
                "next_batch_acc1": eval_next_acc1,
                "next_batch_acc5": eval_next_acc5,
                "next_batch_pred": eval_next_pred,
                "next_batch_target": eval_next_target,
                # "next_batch_pred_entropy": eval_next_pred_entropy,s
                "next_batch_top1_softmax_values": eval_next_top1_softmax_values,
            }
        self.train()
        self.unfreeze()
        # self.quick_train()
        self.log("next_batch_acc1_current", eval_next_acc1, sync_dist=True)
        self.log("next_batch_acc5_current", eval_next_acc5, sync_dist=True)

        self.next_batch_acc1.update(eval_next_acc1)
        next_batch_acc1_online = self.next_batch_acc1.compute()
        self.log("next_batch_acc1", next_batch_acc1_online, sync_dist=True, prog_bar=True)

        # for compatibility with old experiments
        self.log("next_batch_acc1_step", next_batch_acc1_online, sync_dist=True)

        # log_scalars = {
        #     "next_batch_pred_entropy": eval_next_pred_entropy,
        #     # ...
        # }
        # self.log_dict(log_scalars, sync_dist=True, prog_bar=False)

        self.log_histogram({
            "next_batch_pred": eval_next_pred,
            "next_batch_target": eval_next_target,
            "next_batch_top1_softmax_values": eval_next_top1_softmax_values,
            "correct_top1_softmax_values": correct_top1_softmax_values,
            "incorrect_top1_softmax_values": incorrect_top1_softmax_values,
        })

        return out


    def copy_unsup_backbone_to_sup_backbone(self):
        self.supervised_backbone.load_state_dict(self.backbone.state_dict())

    def supervised_step(self, supervised_data, batch_idx, quiet=False):
        #############################################
        # COPY UNSUP MODEL TO SUPERVISED MODEL IF NEEDED
        #############################################
        if not self.share_backbone:
            if "backbone" in self.supervised_param_names and self.num_unsupervised > 0:
                # check if it is the first iteration of the supervised steps
                first_sup_idx_in_sequence = self.step_sequence.index("S")
                is_first_sup = batch_idx % self.num_steps_per_timestep == first_sup_idx_in_sequence
                if is_first_sup:
                    self.copy_unsup_backbone_to_sup_backbone()


        #############################################
        # FORWARD PASS
        #############################################
        outs = {}
        idx, X, targets = supervised_data
        assert isinstance(X, torch.Tensor)

        if X.ndim == 5:
            # Shape of X is (batch_size, 2, 3, 224, 224)
            # concat all the buffer data into the batch dim for inference
            X = X.flatten(0, 1)

            # Split the targets according to the different data streams
            targets = targets.flatten(0, 1)
            idx = idx.flatten(0, 1)

        # Forward pass
        outs.update(self.forward(X))
        logits = outs["logits"]

        # TODO: Implement a logit / softmax / argmax logger
        # that doesn't slow down the inference (i.e. only logs every ~100 steps)

        supervised_acc1 = accuracy_at_k(logits, targets, top_k=[1])[0]
        outs["supervised_acc1"] = supervised_acc1

        if (targets.cpu() == idx.cpu()).all():
            print(batch_idx, supervised_data)
            raise RuntimeError("target indices are bollocks")

        if not(targets.min() >= -1 and targets.max() < self.num_classes):
            print(batch_idx, supervised_data)
            raise RuntimeError("target indices are out of range")
        # assert targets.min() >= -1 and targets.max() < self.num_classes

        # The targets might have been padded with -1 in the first steps when
        # no supervision was available. We need to ignore these values in the loss.
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        loss = loss.nan_to_num()

        outs["supervised_loss"] = loss
        if not quiet:
            self.log("supervised_loss", loss, sync_dist=True)
        return outs


    def unsupervised_step(self, unsupervised_data, batch_idx):
        """
        Wrapper around _unsupervised_step that also logs the metrics.
        """

        #############################################
        # FORWARD PASS
        #############################################
        outs = {}
        _, X, _ = unsupervised_data
        X = [X] if isinstance(X, torch.Tensor) else X
        # check that we received the desired number of crops
        assert len(X) == self.num_crops, f"Expected {self.num_crops} crops, got {len(X)}"

        flattened_X = []
        for x in X:
            if x.ndim == 5:
                # Shape of X is (batch_size, 2, 3, 224, 224)
                #                            ^-- num_buffers
                # concat all the buffer data into the batch dim for inference
                flattened_X.append(x.flatten(0, 1))
            else:
                # Shape of X is (batch_size, 3, 224, 224)
                flattened_X.append(x)
        X = flattened_X

        unsupervised_outs = [self(x) for x in X[: self.num_large_crops]]
        # outs = {k: [out[k] for out in outs] for k in unsupervised_outs[0].keys()}
        for k in unsupervised_outs[0].keys():
            outs[k] = [out[k] for out in unsupervised_outs]

        if self.multicrop:
            multicrop_unsupervised_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_unsupervised_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        assert len(outs["feats"]) == self.num_crops
        return outs

    def _get_step_type(self, batch_idx):
        sequence_idx = batch_idx % self.num_steps_per_timestep
        step_type = self.step_sequence[sequence_idx]
        return step_type


    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """
        iteration_start = time.time()
        timestep = batch_idx // self.num_steps_per_timestep
        sequence_idx = batch_idx % self.num_steps_per_timestep
        step_type = self.step_sequence[sequence_idx]

        self.timestep = timestep
        self.sequence_idx = sequence_idx
        self.step_type = step_type

        if timestep < 100:
            for metric in self.time_metrics.values():
                metric.reset()

        self.data_time.update(time.time() - self.data_start)
        iter_start = time.time()

        logged_metrics = {
            "timestep_step": timestep, # for compatibility with older experiments
            "batch_idx": batch_idx,
            "data_time": self.data_time.compute(),
        }

        #############################################
        # CONDITIONAL FORWARD PASS
        #############################################
        outs = {}
        # Eval step
        if step_type == "E":
            eval_start = time.time()
            outs.update(self.eval_step(batch, batch_idx))
            self.eval_time.update(time.time() - eval_start)
            logged_metrics["eval_time"] = self.eval_time.compute()

        # Unsupervised training
        if step_type == "U":
            unsup_start = time.time()
            outs.update(self.unsupervised_step(batch, batch_idx))
            self.unsup_forward_time.update(time.time() - unsup_start)
            logged_metrics["unsup_forward_time"] = self.unsup_forward_time.compute()

        # Supervised training
        if step_type == "S" and timestep >= self.online_delay:
            sup_start = time.time()
            outs.update(self.supervised_step(batch, batch_idx))
            self.sup_forward_time.update(time.time() - sup_start)
            logged_metrics["sup_forward_time"] = self.sup_forward_time.compute()


        backward_and_update_start = time.time()
        self._manual_backward_and_update(outs, batch_idx)
        self.backward_and_update_time.update(time.time() - backward_and_update_start)
        logged_metrics["backward_and_update_time"] = self.backward_and_update_time.compute()


        self.iter_time.update(time.time() - iter_start)
        logged_metrics["iter_time"] = self.iter_time.compute()
        eta = self.iter_time.compute() * (self.trainer.max_steps - batch_idx)
        logged_metrics["eta"] = eta // 60 # in minutes

        self.log_dict(logged_metrics, on_epoch=False, on_step=True, prog_bar=False)
        self.log("timestep", timestep, on_epoch=False, on_step=True, prog_bar=True)

        self.data_start = time.time()
        self.iter_time.update(time.time() - iteration_start)
        remaining_batches = (self.trainer.estimated_stepping_batches - batch_idx) / 3600
        eta = self.iter_time.compute() * remaining_batches
        self.log("iter_time", self.iter_time.compute(), on_epoch=False, on_step=True)
        self.log("ETA", eta, on_epoch=False, on_step=True)

        return outs


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

            unsup_opt.zero_grad()
            self.manual_backward(outputs["unsupervised_loss"])
            unsup_opt.step()
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

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: List[Any], batch_idx: int) -> None:
        """
        Checking if update call has been made in the `training_step`
        """
        assert self._updated == True, f"Update call has not been made in the {batch_idx} step"
        self._updated = False


class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Extra cfg settings:
            momentum:
                base_tau (float): base value of the weighting decrease coefficient in [0,1].
                final_tau (float): final value of the weighting decrease coefficient in [0,1].
                classifier (bool): whether or not to train a classifier on top of the momentum backbone.
        """

        super().__init__(cfg)

        # initialize momentum timer
        self.momentum_backbone_forward_time = torchmetrics.MeanMetric()

        # make sure that metrics are saved with the checkpoint as part of the state_dict
        self.momentum_backbone_forward_time.persistent(True)

        # initialize momentum backbone
        kwargs = self.backbone_args.copy()

        method: str = cfg.method
        self.momentum_backbone: nn.Module = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            # remove fc layer
            self.momentum_backbone.fc = nn.Identity()

        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # momentum classifier
        if cfg.momentum.classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(cfg.momentum.base_tau, cfg.momentum.final_tau)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.backbone, self.momentum_backbone)]

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(BaseMomentumMethod, BaseMomentumMethod).add_and_assert_specific_cfg(cfg)

        cfg.momentum.base_tau = omegaconf_select(cfg, "momentum.base_tau", 0.99)
        cfg.momentum.final_tau = omegaconf_select(cfg, "momentum.final_tau", 1.0)
        cfg.momentum.classifier = omegaconf_select(cfg, "momentum.classifier", False)

        return cfg

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.momentum_backbone(X)
        out = {"feats": feats}
        if self.momentum_classifier is not None:
            logits = self.momentum_classifier(feats)
            out["logits"] = logits
        return out

    def unsupervised_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for the momentum backbone and optionally the momentum classifier.

        Args:
            batch (List[Any]): list containing [X, targets].
            batch_idx (int): batch index.
        """
        outs = super().unsupervised_step(batch, batch_idx)

        momentum_backbone_forward_start = time.time()

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        X = [x.flatten(0, 1) for x in X]
        targets = targets.flatten(0, 1)

        # remove small crops
        X = X[: self.num_large_crops]


        momentum_outs = [self.momentum_forward(x) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }
        outs.update(momentum_outs)

        timestep = batch_idx // self.num_steps_per_timestep
        if timestep > 100:
            self.momentum_backbone_forward_time.update(time.time() - momentum_backbone_forward_start)
            t = self.momentum_backbone_forward_time.compute()
            self.log_dict({"momentum_backbone_forward_time": t}, sync_dist=True)

        # check if self is exactly the BaseMomentumMethod class
        if self.__class__.__name__ == "BaseMomentumMethod":
            raise ValueError("BaseMomentumMethod should not be used directly.")


        self.data_start = time.time()
        return outs


    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step
