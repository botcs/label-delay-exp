from torch.utils.data import Dataset
import torch

class BatchRepeatDataset(Dataset):
    def __init__(self, dataset, num_repeats, batch_size):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.batch_size = batch_size


        # Implement this in a stateless way:
        #
        #
        # # remove the last batch if it is not full
        # N = len(self.dataset) // self.batch_size * self.batch_size 

        # self.repeat_indices = torch.tile(
        #     torch.arange(len(self.dataset)), 
        #     self.num_repeats
        # )

    def __len__(self):
        return len(self.dataset) * self.num_repeats
    
    def __getitem__(self, index):
        num_repeats = self.num_repeats
        batch_size = self.batch_size
        
        # example for num_repeats = 3 and batch_size = 4:
        # 0123 4567 -> 0123 0123 0123 4567 4567 4567

        # index // batch_size normalizes the index to the batch
        # .. // num_repeats normalizes with the repeats
        # .. * batch_size selects the first index of the batch
        # .. + index % batch_size selects the index within the batch
        repeat_index = index // batch_size // num_repeats * batch_size + index % batch_size
        return self.dataset[repeat_index]