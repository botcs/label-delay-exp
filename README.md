# Label Delay in Continual Learning
Supplementary material for our [NeurIPS paper](https://openreview.net/forum?id=m5CAnUui0Z)
For the browser based demo, please visit the landing page: https://botcs.github.io/label-delay/

This quick intro will help setting up our experimental framework to reproduce our results on the smallest dataset.

Please note that the code is just for the rebuttal period only, upon acceptance the authors will remove the experimental features
and provide a list of scripts to reproduce any result from the manuscript.

### Requirements
1. working conda environment
1. 1GB space for the Yearbook dataset by Ginosar et al.
1. Active Weights and Biases account (https://wandb.ai/) for logging and reporting the results.
1. (Optional but strongly preferred) CUDA compatible hardware

### Setup
1. To set the experimental environment up, run the following
```
conda env create -f environment.yaml
conda activate label-delay
```

1. To download the smallest dataset, Yearbook (N=37,921), run:
```
wget -O faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz "https://www.dropbox.com/scl/fi/7dv71y3nxrcdrpmwntr8e/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?rlkey=h03r92h1mdr9yet2tkqosqq1k&dl=1"

tar -xzvf faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz
mkdir ./datasets/cldatasets/YEARBOOK/ -p
mv faces_aligned_small_mirrored_co_aligned_cropped_cleaned datasets/cldatasets/YEARBOOK/

wget https://raw.githubusercontent.com/katerakelly/yearbook-dating/master/data/faces/men/train.txt -O datasets/cldatasets/YEARBOOK/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/train_M.txt

wget https://raw.githubusercontent.com/katerakelly/yearbook-dating/master/data/faces/men/test.txt -O datasets/cldatasets/YEARBOOK/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/test_M.txt

```

1. Run the training
```
python main_delay.py --config-path=scripts/custom --config-name=iwm.yaml data.dataset=yearbook online.delay=50 online.num_supervised=16 online.sup_buffer_size=524288
```

### Credits
These open source projects played a pivotal role in achieving the results reported in the manuscript:

- https://github.com/vturrisi/solo-learn
- https://github.com/ContinualAI/avalanche
- https://github.com/drimpossible/EvalOCL
- https://github.com/hammoudhasan/CLDatasets
- https://github.com/SchedMD/slurm
- https://github.com/pytorch/pytorch
- and many more...

### Citation
_Update is coming with the NeurIPS '24 proceedings citation_

@article{csaba2023label,
  title={Label Delay in Continual Learning},
  author={Csaba, Botos and Zhang, Wenxuan and M{\"u}ller, Matthias and Lim, Ser-Nam and Elhoseiny, Mohamed and Torr, Philip and Bibi, Adel},
  journal={arXiv preprint arXiv:2312.00923},
  year={2023}
}

### Acknowledgement
We thank the reviewers for investing time into examining our codebase and reproducing our experimental results.
