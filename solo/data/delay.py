import torch 
from torch.utils.data import DataLoader, Dataset, SequentialSampler
# import pytorch_lightning as pl
import lightning.pytorch as pl
from typing import Any, DefaultDict, Dict, Generator, List, Optional, overload, Tuple, Union

class SequentialSamplerWithResume(SequentialSampler):
    """
    Identical to SequentialSampler but starts from an offset.
    """
    def __init__(self, data_source: Dataset, offset: int = 0):
        super().__init__(data_source)
        self._offset = offset

    def __iter__(self) -> Generator[int, None, None]:
        return iter(range(self._offset, len(self.data_source)))
        

class SequentialDataLoader(pl.LightningDataModule):
    """
    When multiple GPU is used use this
    """
    def __init__(
        self,
        **dataloader_kwargs
    ):
        super().__init__()
        self.dataloader_kwargs = dataloader_kwargs


    def prepare_data_per_node(self):
        pass

    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        ret = DataLoader(
            **self.dataloader_kwargs
        )
        return ret

    # def val_dataloader(self):
    #     return None

    # def test_dataloader(self):
    #     return None

    

def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, shuffle: bool = True,
    sampler=None, batch_sampler=None,
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader

