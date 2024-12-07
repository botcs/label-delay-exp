import torch
from torch.utils.data import Dataset
from collections import Counter


class OnlineDatasetWithDelay(Dataset):
    """
    This dataset wraps a generic PyTorch dataset and adds a specified delay to the stream.

    usually for each index a dataset is expected to return a tuple of (data, target) or (data, target, index)

    After the delay is added we have four entities returned by the wrapper
    - test: the most recently observed datapoint, on which the model is immediately evaluated
    - unsup_buffer: a buffer of unlabeled datapoints that the model can use to update its parameters
    - train: the most recently labeled datapoint that the model is allowed to use to update its parameters
    - sup_buffer: a memory replay buffer of labeled datapoints that the model can use to update its parameters

    The wrapper is expected to be used in the following way:
    1 the model receives the test data
    2 the model makes an inference on the test data
    3 the model is evaluated on the test data
    4 the model parameters are updated on:
        - the test data (unsupervised)
        - sample from the unsup_buffer (unsupervised)
        - the train data (supervised)
        - sample from the sup_buffer data (supervised)


    (I know we hate emojis, but this is the best way to explain it)
    The data stream looks like the following:
    🔴 🟠 🟠 🟠 🟠 🟢 🔵 🔵 🔵 🔵 ⚫ ⚫ ⚫ ⚫ ⚫
    ^ the most recent datapoint observed by the model


    Assuming here batch_size=1, the wrapper returns the following data:
    🔴 🟠 🟠 🟠 🟠 🟢 🔵 🔵 🔵 🔵 
    ^ test data (most recently observed data)
    🔴 🟠 🟠 🟠 🟠 🟢 🔵 🔵 🔵 🔵 
          ^ unsup_buffer data (randomly sampled)
    🔴 🟠 🟠 🟠 🟠 🟢 🔵 🔵 🔵 🔵 
                  ^ train data (most recent supervised data)
    🔴 🟠 🟠 🟠 🟠 🟢 🔵 🔵 🔵 🔵 
                        ^ sup_buffer data (randomly sampled)

    This makes up the following batches:
    🔴: test data
    🟠: unsup_buffer data
    🟢: train data
    🔵: sup_buffer data
    --------------------
    ⚫: data that is not used by the model




    The wrapper is stateless, so it can be used in a multiprocessing environment.
    Furthermore the buffer is only realized virtually by using the indices of the dataset.
    """

    def __init__(
        self,
        dataset,
        delay,
        sup_buffer_size,
        eval_transform=None,
        unsupervised_transform=None,
        supervised_transform=None,
        ignore_index=-1,
        hide_labels_in_unsup=True,
        seed=None,
    ):
        self.dataset = dataset
        self.delay = delay
        self.sup_buffer_size = sup_buffer_size
        self.eval_transform = eval_transform
        self.unsupervised_transform = unsupervised_transform
        self.supervised_transform = supervised_transform
        self.ignore_index = ignore_index
        self.hide_labels_in_unsup = hide_labels_in_unsup

        """
        If hide_labels_in_unsup is True, then the labels of the unsupervised data
        will return the ignore_index instead of the actual label.

        Please make sure before running final evaluations that the labels are HIDDEN
        to avoid any leakage of information.
        """

        self.seed = seed
        if seed is not None:
            print(f"Setting manual seed to {seed}")
            torch.manual_seed(seed)


    def concat_streams(self, streams):
        """
        Concatenate the streams of data into a single stream.
        """

        # streams is a list of tuples of (indices, images, targets)
        indices = torch.tensor([stream[0] for stream in streams])

        if isinstance(streams[0][1], torch.Tensor):
            images = torch.stack([stream[1] for stream in streams], dim=0)
        elif isinstance(streams[0][1], list):
            # This is for the case where the dataset returns a list of tensors
            # due to the use of multiple augmentations for contrastive methods
            assert all([len(stream[1]) == len(streams[0][1]) for stream in streams]), \
                "All streams should have the same number of images"

            images = [
                torch.stack([stream[1][i] for stream in streams], dim=0)
                for i in range(len(streams[0][1]))
            ]
        else:
            raise ValueError(f"Unknown type of images: {type(streams[0][1])}")

        # assuming that the targets are ints (not one-hot)
        targets = torch.tensor([stream[2] for stream in streams])

        return indices, images, targets


    def get_eval_data(self, index):
        """
        Get the eval data at the specified index.
        """
        self.dataset.transform = self.eval_transform
        return self.dataset[index]

    def get_supervised_data(self, index):
        """
        Get the supervised data at the specified index.
        """
        eval_index = index
        sup_newest_index = index - self.delay

        self.dataset.transform = self.supervised_transform
        if sup_newest_index < 0:
            # if sup_newest_index < 0, then the train data is not used
            # this happens at the beginning of training
            
            sup_newest_data = sup_buffer_data = self.dataset[0]

            # replace labels with ignore_index
            sup_newest_data = (sup_newest_data[0], sup_newest_data[1], self.ignore_index)
            sup_buffer_data = (sup_buffer_data[0], sup_buffer_data[1], self.ignore_index)

        else:
            # train data
            sup_newest_data = self.dataset[sup_newest_index]

            # sup_buffer data
            start_index = max(sup_newest_index - self.sup_buffer_size, 0)
            end_index = sup_newest_index
            if start_index == end_index:
                sup_buffer_random_index = start_index
            else:
                sup_buffer_random_index = torch.randint(
                    start_index, end_index, [1]
                ).item()
            # sup_buffer_random_index = 0
            sup_buffer_data = self.dataset[sup_buffer_random_index]

        # concatenate train and sup_buffer data
        supervised_data = self.concat_streams([sup_newest_data, sup_buffer_data])

        return supervised_data

    def get_unsupervised_data(self, index):
        """
        Get the unsupervised data at the specified index.
        """
        unsup_newest_index = index
        sup_newest_index = index - self.delay

        #############################
        # UNSUPERVISED DATA
        self.dataset.transform = self.unsupervised_transform

        # sample the most recent datapoint but with the unsupervised transform
        unsup_newest_data = self.dataset[unsup_newest_index]

        if self.delay == 0:
            # if there is no delay, we don't need to sample the unsup_buffer
            unsup_buffer_data = unsup_newest_data
        else:
            # unsup_buffer data
            start_index = max(sup_newest_index, 0)
            end_index = unsup_newest_index

            if start_index == end_index:
                unsup_buffer_random_index = start_index
            else:
                unsup_buffer_random_index = torch.randint(
                    start_index, end_index, [1]
                ).item()
            unsup_buffer_data = self.dataset[unsup_buffer_random_index]

        # set the target of the unsup_buffer data to the ignore_index 
        # to make sure it is not used for training
        if self.hide_labels_in_unsup:
            unsup_newest_data = (unsup_newest_data[0], unsup_newest_data[1], self.ignore_index)
            unsup_buffer_data = (unsup_buffer_data[0], unsup_buffer_data[1], self.ignore_index)

        # concatenate test and unsup_buffer data
        unsupervised_data = self.concat_streams([unsup_newest_data, unsup_buffer_data])

        return unsupervised_data

    def __getitem__(self, index):
        raise NotImplementedError("Please use get_eval_data, get_supervised_data, or get_unsupervised_data")

    def __len__(self):
        return len(self.dataset)


class SupervisionSourceODWD(OnlineDatasetWithDelay):
    """
    Allows picking whether to use newest [N] or random [R]
    for the supervised data.

    if [NN] then the newest (t_0) data and the penultimate (t_-1) data are used
    if [NR] then the newest (t_0) data and a random data point from the buffer are used
    if [RR] then two random data points from the buffer are used

    if [NNN] then the newest (t_0) data, the penultimate (t_-1) data, and the antepenultimate (t_-2) data are used
    if [NNR] then the newest (t_0) data, the penultimate (t_-1) data, and a random data point from the buffer are used
    if [NRR] then the newest (t_0) data, and two random data points from the buffer are used
    if [RRR] then three random data points from the buffer are used

    the ORDERING does NOT matter.
    """

    def __init__(self, *args, **kwargs):
        print(args)
        assert len(args) == 0, "don't pass any positional arguments, use keyword arguments"
        supervision_source = kwargs.pop("supervision_source")
        batch_size = kwargs.pop("batch_size")
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.supervision_source = list(supervision_source)
        if "N" not in self.supervision_source:
            print("WARNING: supervision_source does not contain 'N', \
                this means that the newest data is not used for training")
            print("adding 'N' to supervision_source for sampling only")
            self.supervision_source.append("N")
        
        self.supervision_source = sorted(self.supervision_source)
        assert all([letter in ["N", "R", "W"] for letter in self.supervision_source]), \
            f"supervision_source: {self.supervision_source} contains unknown letters"
        
        print(f"supervision_source: {self.supervision_source}")
            
        # This is not essential
        # if "R" not in self.supervision_source:
        #     print("WARNING: supervision_source does not contain 'R', \
        #         this means that random data is not used for training")
        #     print("adding 'R' to supervision_source for sampling only")
        #     self.supervision_source.append("R")
        
        self.source_counter = Counter(self.supervision_source)
    
    def get_supervised_data(self, index):
        """
        Get the supervised data at the specified index.
        """
        eval_index = index
        sup_newest_index = index - self.delay

        self.dataset.transform = self.supervised_transform


        # if sup_newest_index < 0, then the train data is not used
        # this happens at the beginning of training
        if sup_newest_index < 0:
            # parent class handles this case
            return super().get_supervised_data(index)

        # [NR] Naive scenario: 1:1 ratio of newest and buffer data is used
        if self.supervision_source == ["N", "R"]:
            # parent class handles this case
            return super().get_supervised_data(index)
        

        sources = []
        sanity_check_letters = []
        for source, count in self.source_counter.items():
            if source == "N":
                for i in range(count):
                    idx = max(sup_newest_index - i * self.batch_size, 0)
                    newest_data = self.dataset[idx]
                    sources.append(newest_data)
                    sanity_check_letters.append(source)
            elif source == "R":
                for i in range(count):
                    start_index = max(sup_newest_index - self.sup_buffer_size, 0)
                    end_index = sup_newest_index
                    if start_index == end_index:
                        random_index = start_index
                    else:
                        random_index = torch.randint(
                            start_index, end_index, [1]
                        ).item()
                    random_data = self.dataset[random_index]
                    sources.append(random_data)
                    sanity_check_letters.append(source)
            elif source == "W":
                for i in range(count):
                    # this is ignored as this is not handled by the dataloader
                    sanity_check_letters.append(source)
            else:
                raise ValueError(f"Unknown supervision source: {source}")

        assert self.supervision_source == sanity_check_letters, \
            f"supervision_source: {self.supervision_source} does not match sanity_check_letters: {sanity_check_letters}"

        supervised_data = self.concat_streams(sources)
        return supervised_data
