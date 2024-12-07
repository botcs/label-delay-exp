"""
Implementation of the backward transfer metric.

We load the resnet18 model from a checkpoint and evaluate it on the given 
dataset (CLOC / CGLM).

The evaluation follows Cai et al. (2020) and is done by
applying the classifier in non-overlapping sequential 
batches on the validation set of the dataset. The 
scores are logged in reverse time order.
"""
import sys
import os
import time
sys.path.append(os.getcwd())
import torch
from solo.data.cldataset import H5Dataset
from solo.methods import METHODS
from solo.utils.misc import make_contiguous
from torch.utils.data import DataLoader
from solo.data.classification_dataloader import prepare_transforms
from omegaconf import OmegaConf
import tqdm
import wandb
from solo.data.yearbook import YEARBOOK
from solo.data.fmow import FMOW
import argparse


def main():
    # ckpt_path = sys.argv[1]

    # # runid = ckpt_path.split("/")[-1].split(".")[0]
    # runid = sys.argv[2]
    parser = argparse.ArgumentParser(description='Run ID argparse')
    parser.add_argument('--runid', type=str, help='Run ID')
    args = parser.parse_args()

    runid = args.runid
    ckpt_path = f'trained_models/{runid}/last.ckpt'
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(cfg)
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    raw_cfg["orig_runid"] = runid


    # ============DATA=============
    dataset_name = cfg.data.dataset.upper()
    batch_size = 2**10
    

    _, eval_transform = prepare_transforms(dataset_name.lower())

    if dataset_name.lower() in ['cloc','cglm']:
        dataset = H5Dataset(
            dataset=dataset_name,
            directory=f"{os. path. expanduser('~')}/github/solo-learn/datasets/cldatasets/{dataset_name}/",
            partition="test",
            transform=eval_transform,
        )
    elif dataset_name.lower() == 'yearbook':
        dataset = YEARBOOK(os.path.join(f"~/github/solo-learn/datasets/cldatasets/",  dataset_name.upper()), transform=eval_transform,split='test')
    elif dataset_name.lower() == 'fmow':
        dataset = FMOW(os.path.join(f"~/github/solo-learn/datasets/cldatasets/",  dataset_name.upper()),transform=eval_transform, split='val')
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        drop_last=False,
        prefetch_factor=1,
    )


    # ============MODEL=============
    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    model = model.to(memory_format=torch.channels_last)


    # ============CHECKPOINT=============

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        # model.load_state_dict(checkpoint, strict=True)
        model_state_dict = model.state_dict()
        # check if the model has unmatched / mismatching entries and remove them
        for k, v in model_state_dict.items():
            if k not in checkpoint:
                print(f"Keys not found in checkpoint: {k}")
        for k, v in list(checkpoint.items()):
            if k not in model_state_dict:
                print(f"Removing missing key {k} from checkpoint")
                checkpoint.pop(k)
            
            elif model_state_dict[k].shape != v.shape:
                print(f"Removing mismatching key {k} from checkpoint")
                checkpoint.pop(k)
            
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model from checkpoint")


    # ============EVAL=============
    model.cuda()
    model.eval()
    model.freeze()
    matches = torch.zeros(len(dataset), dtype=torch.bool)
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(dataset))
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits = model.forward(images)["logits"]

            preds = torch.argmax(logits, dim=1)
            from_index = batch_idx*batch_size
            to_index = batch_idx*batch_size+len(images)
            matches[from_index:to_index] = preds == labels
            pbar.update(len(images))


    # ============LOGGING=============
    # compute the backward transfer metric
    # which is a rolling average of the accuracy
    # in reverse time order

    print("Logging backward transfer metric...")
    
    wandb.init(
        project="bwt", 

        name=f"bwt-{cfg.method}-{dataset_name}-{runid}",
        config=raw_cfg
    )

    accuracy_at_time = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        accuracy_at_time[i] = matches[-i:].float().mean()
        if i % 100 == 0:
            wandb.log({
                "backward_transfer": accuracy_at_time[i],
                "timestep": i
            })
            time.sleep(0.01)
        
    


if __name__ == "__main__":
    main()