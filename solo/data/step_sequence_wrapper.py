from torch.utils.data import Dataset
from solo.data.delay_wrapper import OnlineDatasetWithDelay
import torch

class StepSequenceDataset(Dataset):
    def __init__(self, dataset, batch_size, num_supervised, num_unsupervised, category='SSL'):
        assert isinstance(dataset, OnlineDatasetWithDelay)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_supervised = num_supervised
        self.num_unsupervised = num_unsupervised
        if category == "TTA":
            self.step_sequence = "E"  + "S" * self.num_supervised + "U" * self.num_unsupervised
        elif category in ["SSL"]:
            self.step_sequence = "E" + "U" * self.num_unsupervised + "S" * self.num_supervised
        elif category in ["pseudo"]:
            self.step_sequence = "E"
            i = self.num_supervised
            j = self.num_unsupervised
            while i > 0 or j > 0:
                if i > 0:
                    self.step_sequence += "S"
                    i -= 1
                if j > 0:
                    self.step_sequence += "U"
                    j -= 1
        elif category in ["ours", "IWM"]:
            assert self.num_unsupervised == 1
            self.step_sequence = "E" + "U" * self.num_unsupervised + "S" * self.num_supervised
        else:
            raise ValueError(f"Invalid category: {category}")
        self.num_steps_per_timestep = len(self.step_sequence)

        """
        explanation for num_steps_per_timestep:
        1: Evaluation iteration
        self.num_supervised: Supervised iteration(s)
        self.num_unsupervised: Unsupervised iteration(s)


        1. Evaluation (E): take the evaluation batch and compute the online accuracy
                with the most recent backbone + classifier parameters
        2. Unsupervised (U): take the unsupervised batch and compute the unsupervised loss
                with the checkpoint of the backbone from the last (U) update
                followed by that, create a new checkpoint of the backbone
        3. Supervised (S): take the supervised batch and compute the supervised loss
                with the checkpoint of the backbone from the last (U) checkpoint
        """



        # In case of the last batch is < batch_size, 
        # we need to make sure that the indexing is correct
        # which means we need to compute the actual batch size
        self.last_batch_size = len(self.dataset) % self.batch_size

    def __len__(self):
        return len(self.dataset) * self.num_steps_per_timestep
    
    def __getitem__(self, index):
        """
        In case of a batch size of 2, the sequence would look like this:
        received idx:      0  1  2  3  4  5      6  7  8  9 10 11
        required item:    E1 E2 U1 U2 S1 S2     E3 E4 U3 U4 S3 S4

        timestep idx:      0  0  0  0  0  0      1  1  1  1  1  1
        batch idx:         0  0  1  1  2  2      3  3  4  4  5  5
        sequence idx:      0  0  1  1  2  2      0  0  1  1  2  2
        dataset idx:       0  1  0  1  0  1      2  3  2  3  2  3

        """
        
        timestep_idx = index // (self.batch_size * self.num_steps_per_timestep)
        sequence_idx = (index // self.batch_size) % self.num_steps_per_timestep
        step_type = self.step_sequence[sequence_idx]

        # check if the last batch is smaller than batch_size AND if we are in the last batch
        if self.last_batch_size > 0 and timestep_idx == len(self.dataset) // self.batch_size:
            dataset_idx = timestep_idx * self.batch_size + index % self.last_batch_size
        else:
            dataset_idx = timestep_idx * self.batch_size + index % self.batch_size
        
        if step_type == "E":
            return self.dataset.get_eval_data(dataset_idx)
        elif step_type == "U":
            return self.dataset.get_unsupervised_data(dataset_idx)
        elif step_type == "S":
            return self.dataset.get_supervised_data(dataset_idx)
        
        raise ValueError(f"Invalid step: index={index}, timestep_idx={timestep_idx}, sequence_idx={sequence_idx}, step_type={step_type}")