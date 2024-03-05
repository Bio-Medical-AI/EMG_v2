<div align="center">

# EMG Analysis

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-10.1038/srep36571-B31B1B.svg)](https://www.nature.com/articles/srep36571)

</div>

## Description

This project aims to make usage of non-invasive EMG as efficient as possible. **EMG (Electromyographic) signals** are probably the easiest kind of signals from human body to obtain, as they don’t require invasive methods of implanting recording devices. In the same time they are susceptible to noise from electrical devices and skin. It makes usage of EMG signals not efficient enough to use them in commercial products. We want to solve this problem by applying multiple deep learning approaches to it.

### Installation

#### Pip

```bash
# clone project
git clone https://github.com/Bio-Medical-AI/EMG_v2.git
cd EMG_v2

# [OPTIONAL] create conda environment
conda create -n emg_env python=3.10
conda activate emg_env

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/Bio-Medical-AI/EMG_v2.git
cd EMG_v2

# create conda environment and install dependencies
conda env create -f environment.yaml -n emg_env

# activate conda environment
conda activate emg_env
```

## How to download dataset

```bash
dvc pull [path to .dvc file]
```

example

```bash
dvc pull storage/CapgMyo.dvc
```

## How to prepare dataset for work

```bash
python -m download.data_import name=[Name of Dataset]
```

example

```bash
python -m download.data_import name=CapgMyo
```

## How to run

Train model with default configuration. You must be in **source** directory.

```bash
# train on CPU
python -m classification.experiment.base_experiment trainer=cpu

# train on GPU
python -m classification.experiment.base_experiment trainer=gpu
```
