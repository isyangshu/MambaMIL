# MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/isyangshu/MambaMIL?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/isyangshu/MambaMIL?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/isyangshu/MambaMIL?style=flat-square)
[![Arxiv Page](https://img.shields.io/badge/Arxiv-2403.06800-red?style=flat-square)](https://arxiv.org/pdf/2403.06800.pdf)

Pre-Print.

**NOTE**: For subsequent updates of the paper, We will update the arixv version in next month.

**NOTE**: The code for survival analysis is released and the code for cancer subtyping is coming soon.

## Installation
* Environment: CUDA 11.8 / Python 3.10
* Create a virtual environment
```shell
> conda create -n mambamil python=3.10 -y
> conda activate mambamil
```
* Install Pytorch 2.0.1
```shell
> pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
> pip install packaging
```
* Install causal-conv1d
```shell
pip install causal-conv1d==1.1.1
```
* Install Mamba
```shell
> git clone git@github.com:isyangshu/MambaMIL.git
> cd mamba
> pip install .
```
* Other requirements
```shell
pip install scikit-survival==0.22.2
pip install pandas==2.2.1
pip install tensorboardx
```

## Repository Details

* `csv`:  Complete Cbioportal files, including the features path and data splits with 5-fold cross-validation. 
* `datasets`: The code for Dataset, you can just replace the path in Line-25.
* `mamba_ssm`: including the original Mamba, Bi-Mamba from Vim and our proposed SRMamba.
* `models`: the model code about SRMamba, similar to [MCAT](https://github.com/mahmoodlab/MCAT).
* `results`: the results on 12 datasets, including BLCA BRCA CESC CRC GBMLGG KIRC LIHC LUAD LUSC PAAD SARC UCEC.

## How to Train
### Prepare your data
#### WSIs
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 pretrained 1024-dim feature for each 512 $\times$ 512 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

### Training

```shell
sh mambamil.sh
```

## Results
Compared to using different hyperparameter for different datasets in the preprint version, in the current version, our MambaMIL can be set with fixed parameters:
```shell
* LR: 2e-5
* Layer: 2
* Rate: 20
```

I show the csv files about results in the `results`.
* Tables

* Results & Pretrained Parameters:
    - [Google Drive](-)
    - [BaiduYunPan(-)]()

## Paper Details

### Abstract

> Multiple Instance Learning (MIL) has emerged as a dominant paradigm to extract discriminative feature representations within Whole Slide Images (WSIs) in computational pathology. Despite driving notable progress, existing MIL approaches suffer from limitations in facilitating comprehensive and efficient interactions among instances, as well as challenges related to time-consuming computations and overfitting. In this paper, we incorporate the Selective Scan Space State Sequential Model (Mamba) in Multiple Instance Learning (MIL) for long sequence modeling with linear complexity, termed as MambaMIL. By inheriting the capability of vanilla Mamba, MambaMIL demonstrates the ability to comprehensively understand and perceive long sequences of instances. Furthermore, we propose the Sequence Reordering Mamba (SR-Mamba) aware of the order and distribution of instances, which exploits the inherent valuable information embedded within the long sequences. With the SR-Mamba as the core component, MambaMIL can effectively capture more discriminative features and mitigate the challenges associated with overfitting and high computational overhead. Extensive experiments on two public challenging tasks across nine diverse datasets demonstrate that our proposed framework performs favorably against state-of-the-art MIL methods. The code is released at https://github.com/isyangshu/MambaMIL.


## BibTeX

```text
@article{yang2024mambamil,
  title={MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology},
  author={Yang, Shu and Wang, Yihui and Chen, Hao},
  journal={arXiv preprint arXiv:2403.06800},
  year={2024}
}
```
