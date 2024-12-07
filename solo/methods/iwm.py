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

from typing import Any, Dict, Sequence, Tuple
import random
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from collections import Counter

# for logging purposes
import torchmetrics

from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.data.pretrain_dataloader import prepare_base_dataset
from solo.utils.iwm_augment import RandomResizedCrop, ToFloatTensor
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# def data_transform(input):

import torch
import random
import torch.nn.functional as F

import tqdm

from torchvision import transforms as T
from solo.utils.iwm_augment import iwm_raw_transform

class IWM(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Importance Weighted Memory

        Extra cfg settings:
        """

        super().__init__(cfg)

        self.scale_loss: float = cfg.method_kwargs.scale_loss
        self.memory_device: str = cfg.method_kwargs.memory_device
        self.norm_feat: bool = cfg.method_kwargs.norm_feat

        self.preprocess = [
            RandomResizedCrop(224, scale=(0.08, 1.), antialias=True),
            ToFloatTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True
            )
        ]

        self.score_function: str = cfg.method_kwargs.score_function
        assert self.score_function in ["inner", "l2"]

        self.selection_function: str = cfg.method_kwargs.selection_function
        assert self.selection_function in [
            "random",
            "softmax",
            "nearest",
            "max_sum_of_similarity",
        ]

        self.assignment_function: str = cfg.method_kwargs.assignment_function
        assert self.assignment_function in [
            "naive",
            "greedy",
            "eps-greedy",
            "hungarian",
            "temperature",
        ]

        self.class_balanced = cfg.method_kwargs.class_balanced
        if self.class_balanced:
            self.num_queue_classes = self.num_classes
            self.queue_size: int = cfg.method_kwargs.queue_size // self.num_classes
        else:
            self.num_queue_classes = 1
            self.queue_size: int = cfg.online.sup_buffer_size

        self.assignment_function_eps: float = cfg.method_kwargs.assignment_function_eps
        self.base_temperature: float = cfg.method_kwargs.base_temperature
        self.final_temperature: float = cfg.method_kwargs.final_temperature
        assert (
            self.assignment_function_eps >= 0.0 and self.assignment_function_eps <= 1.0
        )
        assert (
            self.assignment_function == "eps-greedy"
            or self.assignment_function_eps == 0.0
        )
        assert (
            "W" in cfg.online.supervision_source or self.assignment_function_eps == 0.0
        )

        if self.selection_function == "max_sum_of_similarity":
            # we use topk when using max_sum_of_similarity
            # that already ensures that the number of
            # selected samples is equal to the batch size
            print("Warning: using max_sum_of_similarity with topk")

        self.use_fixed_features: bool = cfg.method_kwargs.use_fixed_features
        self.use_temperature_schedule: bool = cfg.method_kwargs.use_temperature_schedule

        self.fixed_backbone = None
        if self.use_fixed_features:
            kwargs = self.backbone_args.copy()
            self.fixed_backbone: nn.Module = self.base_model(cfg.method, **kwargs)
            if self.backbone_name.startswith("resnet"):
                # remove fc layer
                self.fixed_backbone.fc = nn.Identity()

            self.fixed_backbone.eval()
            for param in self.fixed_backbone.parameters():
                param.requires_grad = False

        self.use_logits_as_features: bool = cfg.method_kwargs.use_logits_as_features

        x_dim = [3, 224, 224]
        if self.use_logits_as_features:
            features_dim = self.num_classes
        else:
            features_dim = self.features_dim

        assert self.num_unsupervised == 1, "IWM only supports one unsupervised step"

        # supervision source
        self.supervision_source = cfg.online.supervision_source
        assert len(self.supervision_source) > 0
        assert set(self.supervision_source).issubset(set("NRW"))
        self.sup_source_count = Counter(self.supervision_source)
        self.alternating_sources = cfg.method_kwargs.alternating_sources

        if len(self.alternating_sources) > 0:
            assert set(self.supervision_source) == set("NRW")
            for sup_source in self.alternating_sources:
                assert set(sup_source).issubset(set("NRW"))
        else:
            self.alternating_sources = None

        # queue
        self.register_buffer(
            "queue_ptr", torch.zeros(self.num_queue_classes, dtype=torch.long)
        )
        self.register_buffer(
            "queue_datasetidx",
            torch.ones([self.num_queue_classes, self.queue_size], dtype=torch.long)
            * -1,
        )
        self.register_buffer(
            "queue_feat",
            torch.zeros(
                [self.num_queue_classes, self.queue_size, features_dim],
                dtype=torch.float,
            ),
        )

        # to keep track how many samples are stored in the queue (<= queue_size)
        self.register_buffer(
            "queue_len", torch.zeros(self.num_queue_classes, dtype=torch.long)
        )
        self.register_buffer(
            "queue_num_seen_samples",
            torch.zeros(self.num_queue_classes, dtype=torch.long),
        )


        # Dataset, Sampler and Dataloader for quick access to the queue
        # self.setup_dataloader()

        # running average for the three losses
        self.newest_supervised_loss_mean = torchmetrics.MeanMetric()
        self.random_supervised_loss_mean = torchmetrics.MeanMetric()
        self.weight_supervised_loss_mean = torchmetrics.MeanMetric()

        self.newest_supervised_loss_mean.persistent(True)
        self.random_supervised_loss_mean.persistent(True)
        self.weight_supervised_loss_mean.persistent(True)

        self.rescale_losses = cfg.method_kwargs.rescale_losses


    @torch.no_grad()
    def recache_queue(self):
        """
        Load data from the dataset to the queue based on the
        current queue indices.

        This is useful for resuming.
        """

        print("!!!!! Recaching queue")
        dataset = self.cfg.data.dataset
        queue_dataset = prepare_base_dataset(
            dataset=dataset,
            train_data_path=self.cfg.data.train_path,
            transform=iwm_raw_transform,
        )


        probe_sample = queue_dataset[0]
        _, X, _ = probe_sample

        self.queue_input = torch.ones(
            [self.num_queue_classes, self.queue_size, * X.shape],
            dtype=torch.float16,
            device='cpu',
            requires_grad=False
        )

        if self.queue_len.sum() == 0:
            return

        pbar = tqdm.tqdm(total=self.queue_len.sum().item(), desc="Recaching queue")

        for classid, class_datasetidxs in enumerate(self.queue_datasetidx):
            if self.queue_len[classid] == 0:
                continue

            for queueidx, datasetidx in enumerate(class_datasetidxs[:self.queue_len[classid]]):
                X = queue_dataset[datasetidx][1]
                feat = self.backbone(
                    X.to(device=self.device, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).detach()
                self.queue_input[classid, queueidx] = X
                self.queue_feat[classid, queueidx] = feat
                pbar.update(1)


    def on_train_start(self):
        super().on_train_start()
        self.recache_queue()


    def update_temperature(self, cur_step: int, max_steps: int):
        if self.use_temperature_schedule:
            # nicked from solo/utils/momentum.py
            self.current_temperature = (
                self.final_temperature
                - (self.final_temperature - self.base_temperature)
                * (math.cos(math.pi * cur_step / max_steps) + 1)
                / 2
            )
        else:
            self.current_temperature = self.base_temperature

        self.log("temperature", self.current_temperature)

    # def setup_dataloader(self):
    #     """
    #     Here we are just mimicking the main dataset preparation pipeline
    #     but skipping the delay wrapper and the step sequence wrapper.

    #     This is just to make sure that we can access the queue samples
    #     with the same augmentation as if they were loaded by the original
    #     dataloader.

    #     We need to reimplement the dataset preparation pipeline here,
    #     because we don't know if the dataset pipeline has been linked to
    #     the model or not during the initialization of the model.
    #     """
    #     dataset = self.cfg.data.dataset

    #     # Transforms
    #     supervised_transform, eval_transform = prepare_transforms(dataset)

    #     # Dataset
    #     self.queue_dataset = prepare_base_dataset(
    #         dataset=dataset,
    #         train_data_path=self.cfg.data.train_path,
    #         transform=supervised_transform,
    #     )

    #     # Sampler
    #     self.queue_sampler = CustomSampler(self.queue_dataset)

        # # Dataloader
        # self.queue_dataloader = torch.utils.data.DataLoader(
        #     self.queue_dataset,
        #     batch_size=self.cfg.optimizer.batch_size,
        #     sampler=self.queue_sampler,
        #     # num_workers=self.cfg.data.num_workers,
        #     # num_workers=2,
        #     num_workers=self.cfg.method_kwargs.num_workers,
        #     prefetch_factor=self.cfg.method_kwargs.prefetch_factor,
        #     timeout=0,
        #     pin_memory=False,
        #     drop_last=False,
        # )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        cfg = super(IWM, IWM).add_and_assert_specific_cfg(cfg)

        cfg.method_kwargs.queue_size = omegaconf_select(
            cfg, "method_kwargs.queue_size", int(2**16)
        )
        cfg.method_kwargs.memory_device = omegaconf_select(
            cfg, "method_kwargs.memory_device", "cuda"
        )

        cfg.method_kwargs.score_function = omegaconf_select(
            cfg, "method_kwargs.score_function", "inner"
        )
        cfg.method_kwargs.selection_function = omegaconf_select(
            cfg, "method_kwargs.selection_function", "nearest"
        )
        cfg.method_kwargs.assignment_function = omegaconf_select(
            cfg, "method_kwargs.assignment_function", "eps-greedy"
        )
        cfg.method_kwargs.assignment_function_eps = omegaconf_select(
            cfg, "method_kwargs.assignment_function_eps", 0.5
        )
        cfg.method_kwargs.base_temperature = omegaconf_select(
            cfg, "method_kwargs.base_temperature", 10.0
        )
        cfg.method_kwargs.final_temperature = omegaconf_select(
            cfg, "method_kwargs.final_temperature", 0.1
        )

        cfg.method_kwargs.alternating_sources = omegaconf_select(
            cfg, "method_kwargs.alternating_sources", None
        )

        return cfg

    @torch.no_grad()
    def dequeue_and_enqueue(self, datasetidx, feat, X, Y):
        # # Terminology:
        # # the queue_size is the maximum number of samples that can be stored in the queue
        # # the queue_len is the current number of samples that are stored in the queue

        unique_classids = torch.unique(Y)
        # we iterate through the unique class ids to make sure that
        # the samples of each class are stored in a contiguous block
        for unique_classid in unique_classids:
            classid_mask = Y == unique_classid
            classid_datasetidx = datasetidx[classid_mask]
            classid_feat = feat[classid_mask]
            classid_input = X[classid_mask]

            classid_queue_ptr = self.queue_ptr[unique_classid]
            classid_queue_len = self.queue_len[unique_classid]

            # classid_queue_size = self.queue_size // self.num_queue_classes

            # the queue size is already normalized by the number of classes
            classid_queue_size = self.queue_size

            classid_num_samples = classid_datasetidx.shape[0]

            # crop the last batch to avoid the case
            # where the batch size is not a multiple of the queue size
            if classid_queue_ptr + classid_num_samples > classid_queue_size:
                classid_datasetidx = classid_datasetidx[
                    : classid_queue_size - classid_queue_ptr
                ]
                classid_feat = classid_feat[: classid_queue_size - classid_queue_ptr]
                classid_input = classid_input[: classid_queue_size - classid_queue_ptr].detach().cpu()


            self.queue_datasetidx[
                unique_classid,
                classid_queue_ptr : classid_queue_ptr + classid_num_samples,
            ] = classid_datasetidx


            self.queue_feat[
                unique_classid,
                classid_queue_ptr : classid_queue_ptr + classid_num_samples,
            ] = classid_feat
            self.queue_input[
                unique_classid,
                classid_queue_ptr : classid_queue_ptr + classid_num_samples,
            ] = classid_input.to(torch.float16)

            classid_queue_ptr = (
                classid_queue_ptr + classid_num_samples
            ) % classid_queue_size
            classid_queue_len = min(
                classid_queue_len + classid_num_samples, classid_queue_size
            )

            self.queue_ptr[unique_classid] = classid_queue_ptr
            self.queue_len[unique_classid] = classid_queue_len
            self.queue_num_seen_samples[unique_classid] += classid_num_samples

    @torch.no_grad()
    def load_W_data(self, device):
        """
        Args:
            iwm_idx (torch.Tensor): indices of the samples in the queue

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                dataset indices, images and labels of the samples in the queue
        """

        if (not hasattr(self, "iwm_idx")) or (not hasattr(self,"queue_input") ):
            return None, None, None
        if self.iwm_idx.unique(dim=0).numel() == 1:
            return None, None, None

        if (not hasattr(self, "current_sup_step")):
            self.current_sup_step = 0


        if (not hasattr(self, "iwm_X")):
            image_shape = self.queue_input.shape[2:]
            multibatch_size = len(self.iwm_idx)
            self.iwm_datasetidx = torch.empty(
                [multibatch_size],
                device="cpu",
                dtype=torch.long
            )
            self.iwm_X = torch.empty(
                [multibatch_size,*image_shape],
                device=device,
                dtype=torch.uint8
            )

        if self.current_sup_step == 0:
            iwm_idx = self.iwm_idx.cpu()
            self.iwm_datasetidx.copy_(self.queue_datasetidx[iwm_idx[:,0], iwm_idx[:,1]])
            self.iwm_X.copy_(self.queue_input[iwm_idx[:,0], iwm_idx[:,1]])

        batch_start = self.current_sup_step * self.batch_size
        batch_end = (self.current_sup_step + 1) * self.batch_size

        curr_iwm_datasetidx = self.iwm_datasetidx[batch_start:batch_end]
        curr_X = self.iwm_X[batch_start:batch_end]
        curr_Y = self.iwm_idx[batch_start:batch_end, 0]

        self.current_sup_step = (self.current_sup_step + 1)%self.num_supervised


        return curr_iwm_datasetidx,curr_X,curr_Y

    @torch.no_grad()
    def find_nn(self, feat, logit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits is a tensor of shape [batch_size, num_classes]
        # we sample a single classid for each sample in the batch
        # using the logits as probabilities for a multinomial distribution

        # once the classid is sampled, we select a random sample from the queue
        # that has the same classid, either using the nearest neighbor among
        # the samples with the same classid or a random sample among the samples
        # with the same classid

        # the output is a tensor of shape [batch_size] containing the indices
        # of the selected samples in the dataset.
        batch_size = feat.shape[0]
        num_W = max(self.sup_source_count["W"], 1)
        num_supervised = self.num_supervised
        # datasetidx = torch.zeros(
        #     batch_size * num_W * num_supervised, dtype=torch.long, device=feat.device
        # )
        iwm_idx = torch.zeros(
            [batch_size * num_W * num_supervised, 2],
            dtype=torch.long,
            device=feat.device
        )

        # if it's the first iteration just return blank samples
        if self.queue_len.sum() == 0:
            return iwm_idx

        classids = torch.multinomial(
            torch.softmax(logit / self.current_temperature, dim=-1),
            num_W * num_supervised,
            replacement=True,
        )
        classids = classids.flatten(0, 1)


        if (
            self.timestep % self.cfg.log_histogram_every_n_steps == 0
            and self.queue_len.sum() > batch_size
        ):
            # do very detailed logging of the sampling process
            self.log_histogram(
                {
                    "iwm/logit": logit,
                    "iwm/classids": classids,
                }
            )
            self.log("iwm/classids/num_unique_sampled", len(torch.unique(classids)))
            self.log("iwm/logit/std", logit.std())
            self.log("iwm/logit/mean", logit.mean())

            score_stack = torch.ones(
                (len(classids), self.queue_len.max()), device=logit.device
            ) * 0.0

        # iterate over the classids and select a sample for each classid
        for i, classid in enumerate(classids):
            # check if the queue contains samples with the same classid
            if self.queue_len[classid] == 0:
                # if not, set to (0,0) and replace the repeated samples later

                iwm_idx[i, 0] = 0
                iwm_idx[i, 1] = 0
                continue

            classid_queue_feat = self.queue_feat[classid, : self.queue_len[classid]]
            classid_queue_feat = classid_queue_feat / classid_queue_feat.norm(
                dim=-1, keepdim=True
            )
            classid_queue_feat = classid_queue_feat
            classid_feat = feat[i % len(feat)].unsqueeze(0)
            classid_score = F.cosine_similarity(classid_queue_feat, classid_feat)

            if (
                self.timestep % self.cfg.log_histogram_every_n_steps == 0
                and self.queue_len.sum() > batch_size
            ):
                score_stack[i, : self.queue_len[classid]] = classid_score.detach()

            if self.selection_function == "softmax":
                # select a sample using the softmax distribution
                # over the scores of the samples with the same classid
                classid_probs = classid_score.softmax(dim=-1)
                # datasetidx[i] = self.queue_datasetidx[classid][
                #     torch.multinomial(classid_probs, 1)
                # ]
                iwm_idx[i, 0] = classid
                iwm_idx[i, 1] = torch.multinomial(classid_probs, 1)

            elif self.selection_function == "nearest":
                # find the nearest neighbor among the samples with the same classid
                # and select it
                classid_nn_idx = classid_score.argmax()
                # datasetidx[i] = self.queue_datasetidx[classid][classid_nn_idx]
                iwm_idx[i, 0] = classid
                iwm_idx[i, 1] = classid_nn_idx
            elif self.selection_function == "random":
                # select a random sample among the samples with the same classid
                # datasetidx[i] = self.queue_datasetidx[classid][
                #     torch.randint(0, self.queue_len[classid], (1,))
                # ]
                iwm_idx[i, 0] = classid
                iwm_idx[i, 1] = torch.randint(0, self.queue_len[classid], (1,))
            else:
                raise NotImplementedError(
                    f"selection function {self.selection_function} not implemented"
                )

        if (
            self.timestep % self.cfg.log_histogram_every_n_steps == 0
            and self.queue_len.sum() > batch_size
        ):
            # Remove the default values from score_stack for logging
            def rv(x):
                return x[x != 0.0]

            # log the histogram of the scores
            # the `score_stack` has the shape of [batch_size, queue_size], i.e.
            self.log_histogram({"iwm/score": rv(score_stack.flatten())})

            # aggregate the scores across the batch dimension
            # to get the values for each queue (key) position
            self.log_histogram({"iwm/score/key/max": rv(score_stack.max(dim=0).values)})
            self.log_histogram({"iwm/score/key/avg": rv(score_stack.mean(dim=0))})
            self.log_histogram({"iwm/score/key/min": rv(score_stack.min(dim=0).values)})
            self.log_histogram({"iwm/score/key/std": rv(score_stack.std(dim=0))})

            # aggregate the scores across the queue dimension
            # to get the values for each query position
            self.log_histogram({"iwm/score/query/max": rv(score_stack.max(dim=1).values)})
            self.log_histogram({"iwm/score/query/avg": rv(score_stack.mean(dim=1))})
            self.log_histogram({"iwm/score/query/min": rv(score_stack.min(dim=1).values)})
            self.log_histogram({"iwm/score/query/std": rv(score_stack.std(dim=1))})


            # log the query histogram corresponding to the highest/lowest std of the scores
            highest_std_idx = rv(score_stack.std(dim=1)).argmax()
            lowest_std_idx = rv(score_stack.std(dim=1)).argmin()
            self.log_histogram(
                {
                    "iwm/score/query/highest_std": rv(score_stack[highest_std_idx]),
                    "iwm/score/query/lowest_std": rv(score_stack[lowest_std_idx]),
                }
            )


        # The following resampling feature does not improve the performance
        # but to keep the results consistent accross runs we keep it
        # TODO: Remove and cross-validate with newest results

        # go through the datasetidx and check if there are any duplicates
        # if there are duplicates, replace them with random samples
        # ---------------------------------------------
        # note that this is not guaranteed to be unique
        # because the random samples might be the same
        # but it's at least as good as uniform sampling
        iwm_idx_uniques = torch.unique(iwm_idx, dim=0)
        num_random = len(iwm_idx) - len(iwm_idx_uniques)
        if num_random > 0:
            iwm_idx_random = torch.zeros([num_random,2],device=iwm_idx_uniques.device,dtype=torch.long)
            non_zero_ids = torch.nonzero(self.queue_len).squeeze().tolist()
            # tolist returns an int if the list has only one element
            if isinstance(non_zero_ids, int):
                non_zero_ids = [non_zero_ids]
            classids = random.choices(non_zero_ids,k=num_random)

            for i,classid in enumerate(classids):

                iwm_idx_random[i] = torch.tensor([classid, torch.randint(0, self.queue_len[classid], (1,))])



            iwm_idx = torch.cat([iwm_idx_uniques, iwm_idx_random], dim=0)

        self.log("iwm/num_forced_random", num_random)

        # Shuffle the indices to avoid biasing the training
        # towards the first samples in the queue
        iwm_idx = iwm_idx[torch.randperm(len(iwm_idx))]

        return iwm_idx

    @torch.no_grad()
    def load_queue_data(self, device, timestep):
        # log and return one batch of W data
        iwm_datasetidx, iwm_X, iwm_Y = self.load_W_data(device)
        return iwm_datasetidx, iwm_X, iwm_Y

    def supervised_step(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        datasetidx, X, Y = batch
        timestep = self.timestep

        num_newest = self.sup_source_count["N"]
        num_random = self.sup_source_count["R"]
        num_weight = self.sup_source_count["W"]
        num_sources = len(self.supervision_source)

        # separate most recent samples from memory buffer
        if X.ndim == 5:
            if num_newest == 0:
                # this is a hack to make sure that the newest samples
                # are always available for the weighted memory bank
                sup_newest_datasetidx = datasetidx[:, 0]
                sup_newest_X = X[:, 0]
                sup_newest_Y = Y[:, 0]

                sup_random_datasetidx = datasetidx[:, 1 : 1 + num_random].flatten(0, 1)
                sup_random_X = X[:, 1 : 1 + num_random].flatten(0, 1)
                sup_random_Y = Y[:, 1 : 1 + num_random].flatten(0, 1)

            else:
                # the input hasn't been flattened
                sup_newest_datasetidx = datasetidx[:, :num_newest].flatten(0, 1)
                sup_newest_X = X[:, :num_newest].flatten(0, 1)
                sup_newest_Y = Y[:, :num_newest].flatten(0, 1)

                sup_random_datasetidx = datasetidx[
                    :, num_newest : num_newest + num_random
                ].flatten(0, 1)
                sup_random_X = X[:, num_newest : num_newest + num_random].flatten(0, 1)
                sup_random_Y = Y[:, num_newest : num_newest + num_random].flatten(0, 1)

        elif X.ndim == 4:
            # the input has been flattened
            raise NotImplementedError("Flattened input not implemented")
        else:
            raise ValueError(f"Input has invalid shape: {X.shape}")

        sup_weight_datasetidx, sup_weight_X, sup_weight_Y = self.load_queue_data(
            self.device, timestep
        )
        if sup_weight_datasetidx is None:
            sup_weight_datasetidx, sup_weight_X, sup_weight_Y = (
                sup_random_datasetidx,
                sup_random_X,
                sup_random_Y,
            )

        source_data = {
            "N": (sup_newest_datasetidx, sup_newest_X, sup_newest_Y),
            "R": (sup_random_datasetidx, sup_random_X, sup_random_Y),
            "W": (sup_weight_datasetidx, sup_weight_X, sup_weight_Y),
        }
        source_metric = {
            "N": self.newest_supervised_loss_mean,
            "R": self.random_supervised_loss_mean,
            "W": self.weight_supervised_loss_mean,
        }
        abbrev_dict = {"N": "newest", "R": "random", "W": "weight"}
        source_outs = {}

        supervised_loss = 0

        if self.alternating_sources is not None:
            orig_vals = self.supervision_source, self.sup_source_count, num_sources
            self.supervision_source = self.alternating_sources[self.sequence_idx%len(self.alternating_sources)]
            self.sup_source_count = Counter(self.supervision_source)
            num_sources = len(self.supervision_source)

        for source in set(self.supervision_source + "N"):
            source_datasetidx, source_X, source_Y = source_data[source]
            # source_X = self.preprocess(source_X)
            source_X = self.preprocess[0](source_X)
            source_X = self.preprocess[1](source_X)
            source_X = self.preprocess[2](source_X)
            source_X = self.preprocess[3](source_X)

            source_out = super().supervised_step(
                (source_datasetidx, source_X, source_Y), batch_idx, quiet=True
            )
            source_loss_scale = self.sup_source_count[source] / num_sources
            supervised_loss += source_out["supervised_loss"] * source_loss_scale

            source_outs[source] = source_out
            source_metric[source].update(source_out["supervised_loss"].detach())
            self.log_dict(
                {
                    f"{abbrev_dict[source]}_supervised_acc1": source_out[
                        "supervised_acc1"
                    ],
                    f"{abbrev_dict[source]}_supervised_loss": source_out[
                        "supervised_loss"
                    ],
                }
            )

        if self.alternating_sources is not None:
            self.supervision_source, self.sup_source_count, num_sources = orig_vals

        if self.use_fixed_features:
            # use pre-initialized feature extractor
            with torch.no_grad():
                sup_newest_feat = self.fixed_backbone(sup_newest_X)
            if self.use_logits_as_features:
                sup_newest_feat = self.classifier(sup_newest_feat)
        else:
            # no need to unwrap the "multi_crops" wrapper with [0]
            if self.use_logits_as_features:
                sup_newest_feat = source_outs["N"]["logits"]
            else:
                sup_newest_feat = source_outs["N"]["feats"]

        # check if this is the first gradient update iteration at this
        # timestep and if so, add the newest samples to the queue
        first_supervised_iter = self.step_sequence.find("S")

        if self.sequence_idx == first_supervised_iter:
            self.dequeue_and_enqueue(
                datasetidx=sup_newest_datasetidx,
                feat=sup_newest_feat.detach(),
                Y=sup_newest_Y,
                X=sup_newest_X.detach(),
            )
        out = {
            "supervised_loss": supervised_loss,
            "supervised_feats": sup_newest_feat,
        }
        return out


    def unsupervised_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        idx, X, Y = batch
        timestep = self.timestep
        if timestep < self.online_delay:
            return {"unsupervised_loss": torch.tensor(0.0, requires_grad=True)}
        if "W" not in self.supervision_source:
            return {"unsupervised_loss": torch.tensor(0.0, requires_grad=True)}

        assert len(X) == 1, "only one augmentation is supported"
        X = X[0]
        # select the most recent samples
        if X.ndim == 5:
            # the input hasn't been flattened
            unsup_newest_idx = idx[:, 0]
            unsup_newest_X = X[:, 0]
            unsup_newest_Y = Y[:, 0]
        elif X.ndim == 4:
            # the input has been flattened
            batch_size = X.shape[0]
            unsup_newest_idx = idx[: batch_size // 2]
            unsup_newest_X = X[: batch_size // 2]
            unsup_newest_Y = Y[: batch_size // 2]

        unsup_newest_batch = (unsup_newest_idx, [unsup_newest_X], unsup_newest_Y)

        with torch.no_grad():
            sequence_idx = batch_idx % self.num_steps_per_timestep
            # the first iteration is the evaluation
            #   (sequence_idx == 0)
            # the second iteration is the first unsupervised iteration
            #   (sequence_idx == 1)
            if sequence_idx == 1:
                out = super().unsupervised_step(unsup_newest_batch, batch_idx)
                self.unsup_feat = out["feats"][0]
            unsup_feat = self.unsup_feat
            unsup_logit = self.classifier(unsup_feat)

            # Update temperature
            self.update_temperature(
                cur_step=self.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )

            # find nn, init dataloader, only return idx
            self.iwm_idx = self.find_nn(unsup_feat, unsup_logit)
            # save unsupervised Y for later log memory_accuracy
            self.cur_unsupervised_Y = unsup_newest_Y

        # Illustrate the the newest_X and the corresponding iwm_X in wandb
        if timestep % self.cfg.log_images_every_n_steps == 0:
            self.cur_unsupervised_X = unsup_newest_X

        return {"unsupervised_loss": torch.tensor(0.0, requires_grad=True)}

    def log_positive_negative_pairs(
        self, unsup_newest_X, unsup_newest_Y, iwm_X, iwm_Y, num_images=16
    ):
        # find half of the images in the newest_X that have the same label as the iwm_X
        # and the other half that have different labels
        # and log them in wandb

        # find the indices of the images in the newest_X that have the same label as the iwm_X
        # and the other half that have different labels
        same_label_indices = torch.where(unsup_newest_Y == iwm_Y)[0]
        different_label_indices = torch.where(unsup_newest_Y != iwm_Y)[0]

        # narrow the indices to the first num_images
        same_label_indices = same_label_indices[: num_images // 2]
        different_label_indices = different_label_indices[: num_images // 2]

        # select the images from the newest_X that have the same label as the iwm_X
        # and the other half that have different labels
        same_label_X = unsup_newest_X[same_label_indices]
        different_label_X = unsup_newest_X[different_label_indices]

        # select the images from the iwm_X that have the same label as the iwm_X
        # and the other half that have different labels
        same_label_iwm_X = iwm_X[same_label_indices]
        different_label_iwm_X = iwm_X[different_label_indices]

        # select the labels from Y that have the same label as the iwm_Y
        # and the other half that have different labels
        same_label_Y = unsup_newest_Y[same_label_indices]
        different_label_Y = unsup_newest_Y[different_label_indices]
        different_label_iwm_Y = iwm_Y[different_label_indices]

        self.log_corresponding_samples(
            same_label_X,
            same_label_Y,
            same_label_iwm_X,
            same_label_Y,
            figure_title="Correctly labeled memory recalls",
        )
        self.log_corresponding_samples(
            different_label_X,
            different_label_Y,
            different_label_iwm_X,
            different_label_iwm_Y,
            figure_title="Incorrectly labeled memory recalls",
        )

    def log_corresponding_samples(
        self, X, Y, iwm_X, iwm_Y, figure_title="selected corresponding samples"
    ):
        X_pairs = [torch.stack([x1, x2], dim=0) for x1, x2 in zip(X, iwm_X)]
        X_pairs = [
            torchvision.utils.make_grid(xpair, normalize=True, scale_each=True)
            for xpair in X_pairs
        ]
        Y_pairs = list(zip(Y, iwm_Y))
        caption = [f"newest {y1} -> iwm {y2}" for y1, y2 in Y_pairs]

        wandb_logger = self.logger
        # make grid with torchvision
        wandb_logger.log_image(key=figure_title, images=X_pairs, caption=caption)