# SEGNO: Generalizing Equivariant Graph Neural Networks with Physical Inductive Biases

This is the official Pytorch implementation of the paper:  SEGNO: Generalizing Equivariant Graph Neural Networks with Physical Inductive Biases in ICLR 2024.
[**[OpenReview]**](https://openreview.net/forum?id=Hkx1qkrKPr) [**[Paper]**](https://openreview.net/pdf?id=Hkx1qkrKPr) [**[Poster]**](assets/poster.pdf)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hanjq17/GMN/blob/main/LICENSE)

## Requirement

* Python 3.8.12
<<<<<<< master-roy 040b66e0fc4010a1398c4597f554e616cd585895
* For the other packages, please refer to the `environment.yml`.
  You can just execute following command to create the conda environment.

```
conda env create -f environment.yml
```

## Experiments

### N-body System

#### Data Preparation

Use the script `generate_dataset.py` in `nbody/dataset` to generate the n-body simulation data used in experiments.
To generate the `Charged` dataset:

```
python N-body/nbody/dataset/generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small --save_path N-body/nbody/dataset
```

To generate the `Gravity` dataset:

```
python N-body/nbody/dataset/generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small --save_path N-body/nbody/dataset
```

We also provide the download link of the generated dataset: [N-body dataset](https://drive.google.com/drive/folders/1LUnICAS_d1klyzoPeAv2t2fP196yBNf4?usp=sharing).

#### Reproducing the results in Table 1

We use the ``--target`` option to change the predicted interval, where short/medium/long denote the 1000/1500/2000 ts.

To predict the charged N-body system after 1000ts:

```
python N-body/main.py --epochs=500 --max_samples=3000 --layers=8 --hidden_features=64 --norm=none --batch_size=100 --gpu=1 --weight_decay=1e-12 --target short --dataset_dir dataset --dataset charged
```

To predict the charged N-body system after 1500ts:

```
python N-body/main.py --epochs=500 --max_samples=3000 --layers=8 --hidden_features=64 --norm=none --batch_size=100 --gpu=1 --weight_decay=1e-12 --target medium --dataset_dir dataset_long
```

To predict the gravitational N-body system:

```
python N-body/main.py --epochs=500 --max_samples=3000 --layers=8 --hidden_features=64 --norm=none --batch_size=100 --gpu=1 --weight_decay=1e-12 --target short --dataset_dir datset_gravity
```

### MD22

#### Data Preparation
The MD22 dataset can be downloaded from [MD22](http://www.sgdml.org/). The splits are provided in the MD22 folder.

#### Reproducing the results in Table 3

```
python -u MD22/spatial_graph/main_md22.py --data_dir MD22/spatial_graph/md22 --n_layers 5 --lr 1e-3 --mol <mol>
```
The value of `<mol>` can be:
```
nhme, dha, atat, stachyose, atatcgcg, buckyball, nanotube.
```

### CMU motion capture

#### Data Preparation
The raw data were obtained via [CMU Motion Capture Database](http://mocap.cs.cmu.edu/search.php?subjectnumber=35). The preprocessed dataset as well as the splits are provided in spatial_graph/motion folder.

#### Reproducing the results in Table 4

```
python -u spatial_graph/main_motion.py --config_by_file
```

## Reference

Please kindly cite our paper if you find this paper and the codes helpful.  :)

```
@inproceedings{
yang2024improving,
title={Improving Generalization in Equivariant Graph Neural Networks with Physical Inductive Biases},
author={Yang Liu and Jiashun Cheng and Haihong Zhao and Tingyang Xu and Peilin Zhao and Fugee Tsung and Jia Li and Yu Rong},
booktitle={International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Hkx1qkrKPr](https://openreview.net/forum?id=3oTPsORaDH}
}
```

## Acknowledgement

We use the code from the following repository:

* [EGNN](https://github.com/vgsatorras/egnn)
* [GMN](https://github.com/hanjq17/GMN)


