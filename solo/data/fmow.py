import os
import gdown
import torch
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
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
ID_HELD_OUT = 0.1
def download_gdrive(url, save_path, is_folder):
    """ Download the preprocessed data from Google Drive. """
    if not is_folder:
        gdown.download(url=url, output=save_path, quiet=False)
    else:
        gdown.download_folder(url=url, output=save_path, quiet=False)
def download_fmow(data_dir):
    download_gdrive(
        url='https://drive.google.com/u/0/uc?id=1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3&export=download',
        save_path=os.path.join(data_dir, 'fmow.pkl'),
        is_folder=False
    )
def download_detection(data_dir, dataset_file):

    if os.path.isfile(data_dir):
        raise RuntimeError('Save path should be a directory!')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, dataset_file)):
        pass
    else:
        download_fmow(data_dir)

class FMOW(Dataset):
    # download dataset from this link, it is around 50G
    # https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/
    # unzip the dataset 
    # put it in the correct position
    # train and valid split are provided in the meta data


    def __init__(self, directory,transform=None,split='train'):
        """
        split (str): 'train' for training and 'val' for bwt
        """
        super().__init__()
        directory  = f'{os.path.expanduser("~")}/{directory.strip("~")}'
        self.dir = Path(directory)

 
        self.data_file = f'{str(self)}.pkl'

        download_detection(directory, self.data_file)

        self.transform = transform


        self.num_classes = 62
        self.split = split
        self.get_sorted_index()



    def get_sorted_index(self):
        

        # Replace 'your_csv_file.csv' with the path to your CSV file
        csv_file_path =self.dir/'fmow_v1.1'/'rgb_metadata.csv'
        time_column_name = 'timestamp'  # Replace with the name of your time column

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        df = df[df['split'] == self.split]
        df['order'] = df.index
        # Convert the time column to datetime
        df[time_column_name] = pd.to_datetime(df[time_column_name],format='mixed')

        # Sort the DataFrame by the time column
        self.idx_map = df.sort_values(by=time_column_name)





    def __str__(self):
        return 'fmow'
    
    def get_input(self, idx):
        """Returns x for a given idx."""

        img =  default_loader(self.dir / 'fmow_v1.1' / 'images' / f'rgb_img_{idx}.png')
        return img


    def __getitem__(self, idx):
        index = self.idx_map.iloc[idx]['order']


        img = self.get_input(index)
        label = categories.index(self.idx_map.iloc[idx]['category'])


        image_tensor = self.transform(img ) if self.transform is not None else img
        label_tensor = torch.tensor(label,dtype=torch.long)


        return image_tensor, label_tensor

    def __len__(self):
        return self.idx_map.shape[0]




if __name__ == '__main__':
    dataset = FMOW("~/github/solo-learn/datasets/cldatasets/FMOW")
