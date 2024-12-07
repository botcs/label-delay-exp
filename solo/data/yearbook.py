import os
import pickle
from collections import defaultdict
import gdown
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
RAW_DATA_FOLDER = 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
RESOLUTION = 32
ID_HELD_OUT = 0.1
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pil_loader(path) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    

def generate_train_valid_split(directory,class_name):
    class_dir = os.path.join(directory, class_name)
    files_by_year = defaultdict(list)
    files = [os.path.join(class_name, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    for file in files:
        year = file.split('/')[-1].split('_')[0]
        files_by_year[year].append(file)
    train = []
    test = []
    for year in files_by_year.keys():
        random.shuffle(files_by_year[year])
    split_ratio = 0.75
    for year,yearlist in files_by_year.items():
        split_index = int(len(yearlist) * split_ratio)
        train.extend(yearlist[:split_index])
        test.extend( yearlist[split_index:])

        


    with open(os.path.join(directory, f'train_{class_name}.txt'), 'w') as file:
        for item in train:
            file.write(f"{item}\n")

    with open(os.path.join(directory, f'test_{class_name}.txt'), 'w') as file:
        for item in test:
            file.write(f"{item}\n")




class YEARBOOK(Dataset):
    # manually download dataset and unzip from  https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html
    # train and test lists for female are provided
    # we obtain the train and valid list for male here
    # https://github.com/katerakelly/yearbook-dating/tree/master/data/faces/men


    def __init__(self, directory, transform=None, split='train'):
        """
        split (str): 'train' for training and 'test' for bwt
        """
        super().__init__()
        directory = f'{os.path.expanduser("~")}/{directory.strip("~")}'
        self.directory = f'{directory}/faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
        self.classes = ['F', 'M']
        self.transform = transform
        self.num_classes = 2
        self.split = split
        self.load_file()

        



    def load_file(self):
        self.files_by_year = defaultdict(list)
        # Load and sort files by year, maintaining class labels
        for class_name in self.classes:
            lines_from_file = []
            with open(os.path.join(self.directory,f'{self.split}_{class_name}.txt'), "r") as file:
                for line in file:
                    # Removing the newline character at the end of each line
                    path =line.strip().split(' ')[0]
                    if not path.startswith(f'{class_name}/'):
                        path = f'{class_name}/'+path
                    lines_from_file.append(path)
            # class_dir = os.path.join(self.directory, class_name)
            # files = [os.path.join(class_name, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            for file in lines_from_file:
                year = file.split('/')[-1].split('_')[0]
                self.files_by_year[year].append(file)

        # Shuffle images within each year
        for year in self.files_by_year.keys():
            random.shuffle(self.files_by_year[year])

        # Flatten the dictionary into a list while maintaining year order but shuffled within each year
        self.ordered_files = [file for year in sorted(self.files_by_year.keys()) for file in self.files_by_year[year]]

    
    def __len__(self):
        return len(self.ordered_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.directory, self.ordered_files[idx])
        image = default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        # Extract label from the file path
        label = 1 if 'F' == self.ordered_files[idx][0] else 0

        return image, label


    


if __name__ == '__main__':
    generate_train_valid_split(f"{os. path. expanduser('~')}/github/solo-learn/datasets/cldatasets/YEARBOOK/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/",'F')
    dataset = YEARBOOK("~/github/solo-learn/datasets/cldatasets/YEARBOOK")
    import pdb;pdb.set_trace()
