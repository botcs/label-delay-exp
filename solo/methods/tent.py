import logging
import time
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import copy
# import pytorch_lightning as pl
import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from solo.methods.base import BaseMethod, accuracy_at_k


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class TENT(BaseMethod):
    """
    self.supervised_backbone are continually updated by supervised data. 
    self.backbone are copied from self.supervised_backbone for TTA. 
    self.backbone are always configed with TTA setting
    During TTA step, self.backbone is updated by unsupervised data
    If self.backbone is updated at last timestep, it will be used in current timestep evaluation.
    Otherwise, self.supervised_backbone is used for the current timestep evaluation

    Special notes for TTA methods:
    1. Be careful about cfg.online.supervised_param_names. 
    In TTA, unsupervised data are used to update self.backbone. 
    It makes no sense to use the supervised data to only upate classifier and unsupervised data to update the backbone
    2. During the warmup stage in unsupervised step, we only load self.supervised_backbone to self.backbone
    3. Current TTA methods do not suggest to take mutiple U steps
    4. Some TTA methods are dataset based adaptation (multiple adaptation), named offline TTA, wheares this is one batch adaptation, named online TTA.

    Hyperparameter for TENT:
    cfg.method_kwargs.warmup: warmup timestep where TTA is skipped
    cfg.method_kwargs.scale_loss: scale of the TTA loss 
    self.cfg.method_kwargs.tta_lr: learning rate for TTA parameters. 

    When tuning the hyperparameter, tunning scale_loss and tta_lr is equalevent

    
    """


    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
    #     self.automatic_optimization = False
    #     self.before_tta_checkpoint = None

        self.scale_loss: float = cfg.method_kwargs.scale_loss
        self.tta_warmup = cfg.method_kwargs.warmup
        self.step_sequence = "E" +  "S" * self.num_supervised + "U" * self.num_unsupervised 
        self.num_steps_per_timestep = len(self.step_sequence)
        self.share_backbone = False
        self.configure_tta_model()


    def tta_parameters(self)-> List[Dict[str, Any]]:
        param = []
        for nm, m in self.backbone.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        param.append(
                            {'name': f"{nm}.{np}",
                             "params": p,
                             'lr': self.cfg.method_kwargs.tta_lr
                             }
                        )
        return param

    # @property
    # def learnable_params(self) -> List[Dict[str, Any]]:
    #     """Defines learnable parameters for the base class.

    #     Returns:
    #         List[Dict[str, Any]]:
    #             list of dicts containing learnable parameters and possible settings.
    #     """

    #     return [
    #         # {"name": "all", "params": self.parameters()},
    #         {"name": "backbone", "params": self.backbone.parameters()},
    #         {
    #             "name": "classifier",
    #             "params": self.classifier.parameters(),
    #             "lr": self.classifier_lr,
    #             "weight_decay": 0,
    #         },
    #     ]

    def tta_optimizer(self):
        # return torch.optim.Adam(self.tta_parameters,
        #                         lr=self.cfg.method_kwargs.tent_lr,
        #                         betas=(0.9, 0.999),
        #                         weight_decay=0.0)
        parameter = self.tta_parameters()

        return torch.optim.SGD(
            parameter, 
            lr=self.cfg.method_kwargs.tta_lr, 
            momentum=0.9
        )

    def configure_optimizers(self):

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self.tta_optimizer()

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
        # if self.scheduler.lower() == "none":
        #     optimizer1 = super().configure_optimizers()

        #     optimizer2 = self.tta_optimizer()

        #     return [optimizer1, optimizer2]
        # else:
        #     optimizer1, scheduler1 = super().configure_optimizers()

        #     optimizer2, scheduler2 = super().configure_optimizers()
        #     optimizer2 = self.tta_optimizer()

        #     return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def configure_tta_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.backbone.train()
        # disable grad, to (re-)enable only what tent updates
        self.backbone.requires_grad_(False)
        self.classifier.requires_grad_(False)
        if "projection_head" in self.supervised_param_names:
            self.supervised_projection_head.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                ######!!!!!! NOTICE here. For TTA methods we always use the current batch for mean and var for the BN layers
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None




    # def copy_model(self):
    #     # just keep a copy in memory for branching out parameters for TTA
    #     backbone_state = self.backbone.state_dict()
    #     classifier_state = self.classifier.state_dict()
    #     optimizers_state = [opt.state_dict() for opt in self.optimizers()]
    #     return backbone_state, classifier_state, optimizers_state
        

    # def load_model(self, backbone_state, classifier_state, optm_state):
    #     # load the model after evaluation

    #     # load the running stats of batch norm specifically
    #     # (the load_state_dict() function does not work for this)
    #     for nm, m in self.backbone.named_modules():
    #         if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
    #             m.track_running_stats = True
    #             m.running_mean = backbone_state[nm + ".running_mean"]
    #             m.running_var = backbone_state[nm + ".running_var"]
                
        
    #     # load the rest of the model
    #     self.backbone.load_state_dict(backbone_state, strict=True)
    #     self.classifier.load_state_dict(classifier_state, strict=True)
    #     for opt, state in zip(self.optimizers(), optm_state):
    #         opt.load_state_dict(state)

    def copy_sup_backbone_to_tta_backbone(self):
        self.backbone.load_state_dict(self.supervised_backbone.state_dict(),strict=False)
        # print([(name, "Different weights") for (name,param1), (name2, param2) in zip(self.backbone.named_parameters(), self.supervised_backbone.named_parameters()) if not torch.equal(param1.data, param2.data)])
        
    def unsupervised_step(self, unsupervised_data, batch_idx):
        """
        Wrapper around _unsupervised_step that also logs the metrics.
        """
        # self.supervised_backbone_for_check = copy.deepcopy(self.supervised_backbone)
        # self.classifier_for_check = copy.deepcopy(self.classifier)



        self.configure_tta_model()
        # check if it is the first iteration of the unsupervised steps
        first_sup_idx_in_sequence = self.step_sequence.index("U")
        is_first_sup = batch_idx % self.num_steps_per_timestep == first_sup_idx_in_sequence
        if is_first_sup:
            self.copy_sup_backbone_to_tta_backbone()

        
        

        #############################################
        # FORWARD PASS
        #############################################
        
        outs = {}
        _, X, _ = unsupervised_data
        X = [X] if isinstance(X, torch.Tensor) else X
        # check that we received the desired number of crops
        assert len(X) == 1

        unsupervised_outs = [self.tta_forward(x.flatten(0, 1)) for x in X[: self.num_large_crops]]
        # outs = {k: [out[k] for out in outs] for k in unsupervised_outs[0].keys()}
        for k in unsupervised_outs[0].keys():
            outs[k] = [out[k] for out in unsupervised_outs]

        if self.multicrop:
            multicrop_unsupervised_outs = [self.multicrop_forward(x.flatten(0, 1)) for x in X[self.num_large_crops :]]
            for k in multicrop_unsupervised_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        assert len(outs["unsupervised_feats"]) == self.num_crops
        # if no tta at the current step, only copy the model
        if batch_idx // self.num_steps_per_timestep < self.tta_warmup:

            outs['unsupervised_loss'] = 0 * outs["logits"][0].sum()
            return outs
        self.tta_procedure(outs,X=X)
        return outs



    def tta_procedure(self,outs: dict, **kwargs):
        loss = self.scale_loss * softmax_entropy(outs["logits"][0]).mean(0)
        outs['unsupervised_loss'] = loss
    

    def supervised_step(self, supervised_data, batch_idx):
        #############################################
        # NO NEED TO COPY UNSUP MODEL TO SUPERVISED MODEL 
        #############################################
        


        #############################################
        # FORWARD PASS
        #############################################
        outs = {}
        _, X, targets = supervised_data
        assert isinstance(X, torch.Tensor)
        

        if X.ndim == 5:
            # Shape of X is (batch_size, 2, 3, 224, 224)
            # concat all the buffer data into the batch dim for inference
            X = X.flatten(0, 1)

            # Split the targets according to the different data streams
            targets = targets.flatten(0, 1)

        # Forward pass
        outs.update(self.forward(X))
        logits = outs["logits"]

        # TODO: Implement a logit / softmax / argmax logger
        # that doesn't slow down the inference (i.e. only logs every ~100 steps)

        # The targets might have been padded with -1 in the first steps when 
        # no supervision was available. We need to ignore these values in the loss.
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # print(loss)
        loss = loss.nan_to_num()
        # print (loss)
        outs["supervised_loss"] = loss
        
        self.log("supervised_loss", loss, sync_dist=True)

        return outs

    
    # def unsupervised_step(self, unsupervised_data, batch_idx, logged_metrics, outs):
    #     self.configure_tent_model()
    #     super().unsupervised_step(unsupervised_data, batch_idx, logged_metrics, outs)
    #     loss = self.scale_loss * softmax_entropy(outs["logits"][0]).mean(0)
    #     outs['tent_loss'] = loss
    #     logged_metrics['tent_loss'] = loss.detach().item()

    # def training_step(self, batch: List[Any], batch_idx: int):
    #     opt, opt_tent = self.optimizers()

    #     if self.scheduler.lower() != "none":
    #         sch, sch_tent = self.lr_schedulers()
    #     else:
    #         sch, sch_tent = None, None

    #     eval_data, unsupervised_data, supervised_data = batch
    #     outs = {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
    #     logged_metrics = {}

    #     iter_start = time.time()
    #     # Online Eval
    #     self.eval_step(eval_data, batch_idx, logged_metrics, outs)

    #     # Restore model to train mode
    #     if batch_idx % self.cfg.online.batch_repeat == 0:
    #         if self.before_tta_checkpoint is not None:
    #             self.load_model(*self.before_tta_checkpoint)
    #         self.unfreeze()
    #         self.train()

    #     # Supervised training
    #     self.supervised_step(supervised_data, batch_idx, logged_metrics, outs)
    #     opt.zero_grad()
    #     self.manual_backward(outs["loss"])
    #     opt.step()
    #     if sch is not None:
    #         sch.step()

    #     # test time adaptation
    #     if batch_idx % self.cfg.online.batch_repeat == self.cfg.online.batch_repeat - 1 and logged_metrics['timestep'] > self.tta_warmup:
    #         self.before_tta_checkpoint = self.copy_model()
            
    #         self.unsupervised_step(unsupervised_data, batch_idx, logged_metrics, outs)
    #         opt_tent.zero_grad()
    #         loss = outs['tent_loss']
    #         self.manual_backward(loss)
    #         opt_tent.step()
    #         if sch_tent is not None:
    #             sch_tent.step()
            

    #     self.iter_time.update(time.time() - iter_start)

    #     # Logging
    #     if batch_idx < 100:
    #         # reset timers
    #         self.eval_time.reset()
    #         self.unsup_time.reset()
    #         self.sup_time.reset()
    #         self.iter_time.reset()
    #     else:
    #         logged_metrics.update({
    #             "eval_time": self.eval_time.compute(),
    #             "unsup_time": self.unsup_time.compute(),
    #             "sup_time": self.sup_time.compute(),
    #             "iter_time": self.iter_time.compute(),
    #         })

    #     self.log_dict(logged_metrics, on_epoch=True, sync_dist=True)
    def tta_forward(self, X) -> Dict:
        # Perform the forward pass
        # print([(name, "Different weights") for (name,param1), (name2, param2) in zip(self.backbone.named_parameters(), self.supervised_backbone.named_parameters()) if not torch.equal(param1.data, param2.data)])
        outs = {}
        

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        

        feats = self.backbone(X)
        outs["unsupervised_feats"] = feats

        
        if "projection_head" in self.supervised_param_names:
            outs["unsupervised_feats"] = self.supervised_projection_head(outs["supervised_feats"])    

        
        logits = self.classifier(outs["unsupervised_feats"])
        outs["logits"] = logits
        return outs


    def eval_step(self, eval_data, batch_idx: int) -> Dict:
        """
        THE ONLY CHANGE of this file to the base method is to use self.backbone to forward the eval batch
        """
        ##############################
        # EVALUATION ON NEXT BATCH
        ##############################
        self.eval()
        self.freeze()
        # try:
        #     # print([(name, "Different weights") for (name,param1), (name2, param2) in zip(self.backbone.named_parameters(), self.supervised_backbone_for_check.named_parameters()) if not torch.equal(param1.data, param2.data)])
        #     # print([(name, "Different weights") for (name,param1), (name2, param2) in zip(self.classifier.named_parameters(), self.classifier_for_check.named_parameters()) if not torch.equal(param1.data, param2.data)])
        # except AttributeError:
        #     print ('no saved supervised backbone')
        # self.quick_eval()
        with torch.no_grad():
            eval_ind, eval_x, eval_target = eval_data

            batch_size = len(eval_x)
            eval_next_x = eval_x
            eval_next_target = eval_target
            if batch_idx // self.num_steps_per_timestep < self.tta_warmup:
                
                eval_next_logit = self.forward(eval_next_x)["logits"]
            else:
                # import pdb; pdb.set_trace()
                # print ( (self.supervised_backbone.layer4[0].bn1.weight - self.backbone.layer4[0].bn1.weight).sum())
                eval_next_logit = self.tta_forward(eval_next_x)["logits"]
            
            eval_next_pred = torch.argmax(eval_next_logit, dim=1)
            eval_next_pred_entropy = softmax_entropy(eval_next_logit).mean()
            eval_next_top1_softmax_values, _ = torch.topk(
                eval_next_logit.softmax(dim=1), k=1, dim=1
            )
            if self.num_classes > 5:
                eval_next_acc1, eval_next_acc5 = accuracy_at_k(eval_next_logit, eval_next_target, top_k=[1, 5]) 
            else:
                eval_next_acc1 = accuracy_at_k(eval_next_logit, eval_next_target, top_k=[1])[0]
                eval_next_acc5 = 1.0
            # print(eval_next_acc1)

            out = {
                "next_batch_acc1": eval_next_acc1,
                "next_batch_acc5": eval_next_acc5,
                "next_batch_pred": eval_next_pred,
                "next_batch_target": eval_next_target,
                "next_batch_pred_entropy": eval_next_pred_entropy,
                "next_batch_top1_softmax_values": eval_next_top1_softmax_values,
            }
        self.train()
        self.unfreeze()
        
        # self.quick_train()
        self.log("next_batch_acc1_current", eval_next_acc1, on_epoch=False, sync_dist=True)
        self.log("next_batch_acc5_current", eval_next_acc5, on_epoch=False, sync_dist=True)
        
        self.next_batch_acc1.update(eval_next_acc1)
        next_batch_acc1_online = self.next_batch_acc1.compute()
        self.log("next_batch_acc1", next_batch_acc1_online, on_epoch=False, sync_dist=True)

        log_scalars = {
            "next_batch_pred_entropy": eval_next_pred_entropy,
            # ...
        }
        self.log_dict(log_scalars, on_epoch=False, sync_dist=True, prog_bar=False)

        if self.global_step % self.cfg.log_histogram_every_n_steps == 0:
            log_histograms = {
                "next_batch_pred": eval_next_pred.tolist(),
                "next_batch_target": eval_next_target.tolist(),
                "next_batch_top1_softmax_values": eval_next_top1_softmax_values.tolist(),
            }
            # Currently you can't log tensor histograms only scalars in the PL api
            try:
                if any(isinstance(l, pl.loggers.wandb.WandbLogger) for l in self.loggers):
                    wandb = pl.loggers.wandb.wandb
                    # check if wandb was initialized
                    if wandb.run is not None:
                        wandb.log({
                            f"histogram/{k}": wandb.Histogram(v) for k, v in log_histograms.items()
                        })
            except ValueError:
                print(log_histograms)

        return out

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
            if batch_idx // self.num_steps_per_timestep > self.tta_warmup:
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
