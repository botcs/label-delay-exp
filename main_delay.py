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

import inspect
import os
import hydra
import torch
import json
from datetime import timedelta
from omegaconf import DictConfig, OmegaConf


from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch.utilities.model_summary as model_summary
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler
from solo.utils.iwm_augment import iwm_raw_transform

from solo.args.delay import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.classification_dataloader import prepare_transforms
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_datasets,
)
# from torch.utils.data import DataLoader
from solo.data.delay import SequentialDataLoader, SequentialSamplerWithResume

from solo.methods import METHODS
# from solo.utils.auto_resumer import AutoResumer
# from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
from solo.utils.misc import omegaconf_select

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model

    # ============CONFIG=============
    OmegaConf.set_struct(cfg, False)
    # cfg = parse_cfg(cfg)

    # ============RESUME=============
    # Check if resume_wandb_run is passed
    resume_wandb_run = omegaconf_select(cfg, "resume_wandb_run", None)
    ckpt_path = omegaconf_select(cfg, "resume_from_checkpoint", None)

    if resume_wandb_run is not None:
        assert ckpt_path is None, "Cannot resume from both wandb and local checkpoint"
        ckpt_path = f"{cfg.checkpoint.dir}/{resume_wandb_run}/last.ckpt"
    else:
        cfg = parse_cfg(cfg)
        

    if ckpt_path is not None:
        # load the checkpoint
        resume_ckpt = torch.load(ckpt_path, map_location="cpu")
        resume_cfg = resume_ckpt["hyper_parameters"]["cfg"]
        resume_cfg = OmegaConf.create(resume_cfg)

        cfg = resume_cfg

    # ============SEED=============
    seed_everything(cfg.seed)
    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    print("Full config:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae", "base",'tent','sar',"adabn",'eata','eat','cotta',"iwm","pseudo"],\
            "Only WMSE and MAE support num_large_crops != 2"
        
    # ============MODEL=============
    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    print("Model summary:")
    print(model_summary.ModelSummary(model))


    # ============DATA=============
    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    elif cfg.data.dataset in ["cloc", "cglm", "yearbook", 'fmow', "dummy"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format
        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )
    # pretrain dataloader
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        unsupervised_transform = FullTransformPipeline(pipelines)

        # TODO: it's using the precoded val transforms in the `classification_dataloader.py`,
        # but it should use the ones from the config
        supervised_transform, eval_transform = prepare_transforms(cfg.data.dataset)

        if cfg.method == "iwm":
            supervised_transform = iwm_raw_transform

        if cfg.debug_augmentations:
            print("Transforms:")
            print("eval_transform:", eval_transform)
            print("unsupervised_transform:", unsupervised_transform)
            print("supervised_transform:", supervised_transform)

        effective_batch_size = cfg.optimizer.batch_size
        train_dataset = prepare_datasets(
            dataset=cfg.data.dataset,
            eval_transform=eval_transform,
            unsupervised_transform=unsupervised_transform,
            supervised_transform=supervised_transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            sup_buffer_size=cfg.online.sup_buffer_size,
            delay=cfg.online.delay * effective_batch_size,
            num_supervised=cfg.online.num_supervised,
            num_unsupervised=cfg.online.num_unsupervised,
            batch_size=effective_batch_size,
            category=cfg.category,
            supervision_source=cfg.online.supervision_source,
        )

        prefetch_factor = cfg.data.prefetch_factor
        if prefetch_factor == 0:
            prefetch_factor = None
        
        sampler = None
        if ckpt_path is not None:
            resume_batch_state = resume_ckpt["loops"]["fit_loop"]["epoch_loop.batch_progress"]
            resume_batch_started = resume_batch_state["total"]["started"]
            resume_batch_completed =  resume_batch_state["total"]["completed"]
            assert resume_batch_started >= resume_batch_completed,\
                f"resume_batch_started = {resume_batch_started} > resume_batch_completed = {resume_batch_completed}"
            print(f"Resuming from batch_idx = {resume_batch_completed}")

            offset = resume_batch_completed * effective_batch_size
            sampler = SequentialSamplerWithResume(
                train_dataset, offset=offset
            )
        
        train_loader = SequentialDataLoader(
            dataset=train_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            pin_memory=False,
            prefetch_factor=prefetch_factor,
            sampler=sampler,
            drop_last=True,
        )

    # ============CALLBACK=============
    callbacks = []

    # wandb logging
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    cfg.slurm_job_id = slurm_job_id

    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name+slurm_job_id,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if resume_wandb_run is not None else None,
            id=resume_wandb_run,
        )
        wandb_logger.watch(model, log="gradients", log_freq=1000)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

            
    run_id = wandb_logger.experiment.id if cfg.wandb.enabled else f"debug/{cfg.method}"
    run_dir = os.path.join(cfg.checkpoint.dir, run_id)

    if cfg.checkpoint.enabled:
        template_filename = "{method}-slurmid{slurm_id}".format(method=cfg.method, slurm_id=slurm_job_id)
        checkpointer = ModelCheckpoint(
            dirpath=run_dir,
            # train_time_interval=timedelta(seconds=cfg.checkpoint.train_time_interval),
            save_top_k=-1,
            every_n_train_steps=500,
            # monitor="timestep_step",
            # mode="min",
            save_last=True,
            verbose=True,
            filename=template_filename+"-{timestep_step:07.0f}",
        )

        callbacks.append(checkpointer)
        # pass

    

    if cfg.auto_umap.enabled:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)


    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": cfg.checkpoint.enabled,
            "num_sanity_val_steps": 0,
            "default_root_dir": run_dir,
        }
    )
    if cfg.strategy is not None:
        trainer_kwargs["strategy"] = cfg.strategy
        trainer_kwargs["use_distributed_sampler"] = True
    else:
        del(trainer_kwargs["strategy"])

     
    # =============TRAINING=============
    # profiller = SimpleProfiler(filename='time',dirpath='./') 
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=train_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
