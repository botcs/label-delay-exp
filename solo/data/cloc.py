################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
################################################################################

"""CLOC Pytorch Dataset
Installation: https://github.com/IntelLabs/continuallearning/tree/main/CLOC
Paper: http://vladlen.info/papers/CLOC.pdf
"""

# from typing import Any, List
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets.folder import default_loader
import torch
import tqdm

# try importing accimage and set backend if succesful
try:
    import accimage
    import torchvision

    torchvision.set_image_backend("accimage")
except ImportError:
    pass

# https://stackoverflow.com/a/2135920
def split_range(a, n):
    k, m = divmod(len(a), n)
    return (list(a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]) for i in range(n))

class CLOCDataset(Dataset):
    """CLOC Pytorch Dataset
    - Training Images in total: 38,003,083
    - Corss Validation Images in total: 2,064,152
    - Test Images in total: 383,229
    - Shape of images: torch.Size([1, 3, 640, 480])
    """

    splits = ["train", "valid", "test"]

    def __init__(
        self,
        dataset_root,
        metadata_dir="cloc/release/",
        image_dir="cloc/images/",
        split="train",
        transform=ToTensor(),
        target_transform=None,
        debug=False,
        dummy=False,
        cache_num_images=0,
    ):
        super().__init__()
        assert split in self.splits
        self.split = split

        self.transform = transform
        self.target_transform = target_transform

        self.debug = debug
        self.dummy = dummy
        self.cache_num_images = cache_num_images

        if self.dummy:
            return

        assert cache_num_images >= 0

        if cache_num_images == 0:
            abs_image_dir = dataset_root + "/" + image_dir + "/"
        else:
            # the cache metadata holds the absolute path to the images
            # so we don't need to specify the image_dir
            abs_image_dir = ""

        assert self.split in ["train", "valid"]

        if self.split == "train" or self.split == "valid":
            abs_metadata_dir = dataset_root + "/" + metadata_dir + "/"
        else:
            raise NotImplementedError("Test set not implemented yet")

        abs_metadata_dir = os.path.expanduser(abs_metadata_dir)
        abs_image_dir = os.path.expanduser(abs_image_dir)

        self.metadata_dir = abs_metadata_dir
        self.image_dir = abs_image_dir

        self._make_data()
        self._set_data_idx()

        # self.log.info(f"Images in total: {self.__len__()}")

    def _make_data(self):
        # relies on copy_cloc_to_local.py to be run first that caches the data
        cache_suffix = "" if self.cache_num_images == 0 else f"_{self.cache_num_images}"

        if self.split in ["train", "valid"]:
            binary_fnames = [
                f"{self.metadata_dir}/{self.split}{cache_suffix}_labels.torchSave",
                f"{self.metadata_dir}/{self.split}{cache_suffix}_time.torchSave",
                f"{self.metadata_dir}/{self.split}{cache_suffix}_userID.torchSave",
                f"{self.metadata_dir}/{self.split}{cache_suffix}_store_loc.torchSave",
            ]
            bins = []
            for fname in tqdm.tqdm(binary_fnames, desc="Loading CLOC dataset"):
                assert os.path.exists(fname), f"File {fname} does not exist"
                bins.append(torch.load(fname))
            self.labels, self.time_taken, self.user, self.store_loc = bins
            self.store_loc = list(map(lambda s: s.strip(), self.store_loc))



        if self.debug:
            self.labels = self.labels[:10000]
            self.store_loc = self.store_loc[:10000]

    def _set_data_idx(self):
        self.data_size = len(self.labels)
        self.data_idx = list(range(0, self.data_size))

    def get_paths_and_targets(self):
        paths_and_targets = []
        paths_and_targets.append(list(zip(self.store_loc, self.labels, self.data_idx)))
        return paths_and_targets, self.root

    def _dummy_getitem(self, index):
        sample = default_loader("./datboi.jpg")
        target = torch.randint(0, 10, (1,)).item()

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, target


    def __getitem__(self, index):
        if self.dummy:
            return self._dummy_getitem(index)
        index = self.data_idx[index]
        index_pop = index

        target = self.labels[index_pop]
        path = self.image_dir + self.store_loc[index_pop]
        sample = default_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, target

    def __len__(self):
        if self.dummy:
            return 10000
        return len(self.data_idx)
