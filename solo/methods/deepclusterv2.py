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

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.deepclusterv2 import deepclusterv2_loss_func
from solo.methods.base import BaseMethod
from solo.utils.kmeans import KMeans
from solo.utils.misc import omegaconf_select


class DeepClusterV2(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements DeepCluster V2 (https://arxiv.org/abs/2006.09882).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                num_prototypes (Sequence[int]): number of prototypes.
                temperature (float): temperature for the softmax.
                kmeans_iters (int): number of iters for k-means clustering.
                kmeans_freq (int): frequency of k-means clustering.
                scale_loss (float): scale the loss by this factor.
        """

        super().__init__(cfg)

        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.temperature: float = cfg.method_kwargs.temperature
        self.num_prototypes: Sequence[int] = cfg.method_kwargs.num_prototypes
        self.kmeans_iters: int = cfg.method_kwargs.kmeans_iters
        self.kmeans_freq: int = cfg.method_kwargs.kmeans_freq
        self.scale_loss: float = cfg.method_kwargs.scale_loss
        self.warmup_iters: int = cfg.method_kwargs.warmup_iters
        self.projection_memory_size: int = cfg.method_kwargs.projection_memory_size

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # prototypes
        self.prototypes = nn.ModuleList(
            [nn.Linear(proj_output_dim, np, bias=False) for np in self.num_prototypes]
        )

        # normalize and set requires grad to false
        for proto in self.prototypes:
            for params in proto.parameters():
                params.requires_grad = False
            proto.weight.copy_(F.normalize(proto.weight.data.clone(), dim=-1))


        # initialize memory banks
        # size_memory_per_process = len(self.trainer.train_dataloader) * self.batch_size
        self.size_memory_per_process = self.projection_memory_size
        self.register_buffer(
            "local_memory_index",
            torch.zeros(self.size_memory_per_process).long().to(self.device, non_blocking=True),
        )
        self.register_buffer(
            "local_memory_embeddings",
            F.normalize(
                torch.randn(self.num_large_crops, self.size_memory_per_process, self.proj_output_dim),
                dim=-1,
            ).to(self.device, non_blocking=True),
        )

        # [author1]: current issue with the local memory buffer is that when we are using multiple gpus
        # then the local memory buffer is not shared between the gpus
        # e.g:
        # rank 0 batch 129 test_idxs tensor([1032, 1034, 1036, 1038], device='cuda:0') unsup_buff_idxs tensor([1029, 1014, 1027, 1019], device='cuda:0')
        # rank 1 batch 129 test_idxs tensor([1033, 1035, 1037, 1039], device='cuda:1') unsup_buff_idxs tensor([1019, 1026, 1018, 1028], device='cuda:1')

        # this means that for rank 0 we might be missing the odd indices and for rank 1 we might be missing the even indices
        # this won't result in a NaN loss, because the loss is calculated only for the indices that are present in the local memory buffer
        # but can cause discrepancies in the results.

        self.assignments = -torch.ones(
            len(self.num_prototypes), self.projection_memory_size, device=self.device
        ).long()

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(DeepClusterV2, DeepClusterV2).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.temperature = omegaconf_select(cfg, "method_kwargs.temperature", 0.1)
        cfg.method_kwargs.num_prototypes = omegaconf_select(
            cfg,
            "method_kwargs.num_prototypes",
            [3000, 3000, 3000],
        )
        cfg.method_kwargs.kmeans_iters = omegaconf_select(cfg, "method_kwargs.kmeans_iters", 10)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and prototypes parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def on_train_start(self) -> None:
        """Initializes the k-means algorithm."""

        """Gets the world size and initializes the memory banks."""
        #  k-means needs the world size and the dataset size
        self.world_size = self.trainer.world_size if self.trainer else 1

        # build k-means helper object
        self.kmeans = KMeans(
            world_size=self.world_size,
            rank=self.global_rank,
            num_large_crops=self.num_large_crops,
            dataset_size=self.projection_memory_size,
            proj_features_dim=self.proj_output_dim,
            num_prototypes=self.num_prototypes,
            kmeans_iters=self.kmeans_iters,
        )

        self.assignments = self.assignments.to(self.device)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        """Prepares assigments and prototype centroids for the next iteration."""

        if batch_idx % self.kmeans_freq == 0 and batch_idx > self.warmup_iters:
            self.assignments, centroids = self.kmeans.cluster_memory(
                self.local_memory_index, self.local_memory_embeddings
            )


            self.assignments = self.assignments.to(self.device)
            for proto, centro in zip(self.prototypes, centroids):
                # [author1]: had to add this line because
                # the no-grad req is overwritten somewhere
                proto.weight.requires_grad = False
                proto.weight.copy_(centro)

    def update_memory_banks(self, idxs: torch.Tensor, z: torch.Tensor, batch_idx: int) -> None:
        """Updates DeepClusterV2's memory banks of indices and features.

        Args:
            idxs (torch.Tensor): set of indices of the samples of the current batch.
            z (torch.Tensor): projected features of the samples of the current batch.
            batch_idx (int): batch index relative to the current epoch.
        """

        # split idxs and z into buffer and most recent batch
        test_idxs, unsup_buff_idxs = idxs.unbind(dim=1)

        # modulo the indices to make sure they are in the range of the local memory buffer
        test_idxs = test_idxs % self.size_memory_per_process
        unsup_buff_idxs = unsup_buff_idxs % self.size_memory_per_process

        for c, z_c in enumerate(z):
            test_z_c, unsup_buff_z_c = z_c[0::2], z_c[1::2]
            self.local_memory_embeddings[c, test_idxs] = test_z_c.detach()
            self.local_memory_embeddings[c, unsup_buff_idxs] = unsup_buff_z_c.detach()

        self.local_memory_index[test_idxs] = test_idxs

        self.assignments = self.assignments.to(self.device)
        if batch_idx > self.warmup_iters:
            # just infer the assignments for the newest samples without updating the centroids
            for i, proto in enumerate(self.prototypes):
                # use the same rolling indexing as in ./solo/utils/kmeans.py#L167
                c = i % self.num_large_crops
                z_c = z[c]
                test_z_c, unsup_buff_z_c = z_c[0::2], z_c[1::2]
                self.assignments[i, test_idxs] = proto(test_z_c).argmax(dim=-1)
                self.assignments[i, unsup_buff_idxs] = proto(unsup_buff_z_c).argmax(dim=-1)

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the prototypes.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent,
                the projected features and the logits.
        """

        out = super().forward(X)
        z = F.normalize(self.projector(out["feats"]))
        p = torch.stack([p(z) for p in self.prototypes])
        out.update({"z": z, "p": p})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DeepClusterV2 reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DeepClusterV2 loss and classification loss.
        """


        eval_data, unsupervised_data, supervised_data = batch

        # Only use the memory buffer from the unsupervised data
        idxs = unsupervised_data[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]
        p1, p2 = out["p"]
        # the p1 and p2 are telling us which prototype is the closest to the
        # current sample.

        # the shape of p1 and p2 is [len(num_prototypes), 2xbatch_size, num_prototypes]
        # p1 and p2 refer to the two different crops of the same image
        # the 2x batchsize comes from the `test` and `unsup_buffer` samples

        # ------- update memory banks -------
        # [author1]: we need to update the memory banks with the most recent data
        # before we can compute the loss
        self.update_memory_banks(idxs, [z1, z2], batch_idx)

        # ------- deepclusterv2 loss -------
        if batch_idx > self.warmup_iters:
            preds = torch.stack([p1, p2], dim=1)

            # when flattening the idxs we flatten the test and unsup_buffer dimension
            # it's going to be the same ordering as the input images were ordered
            assignments = self.assignments[:, idxs.flatten()%self.size_memory_per_process]
            deepcluster_loss = deepclusterv2_loss_func(preds, assignments, self.temperature)
        else:
            deepcluster_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train_deepcluster_loss", deepcluster_loss, on_epoch=True, sync_dist=True)

        return self.scale_loss * deepcluster_loss + class_loss
