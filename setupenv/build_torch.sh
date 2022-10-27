#!/bin/bash

## build torch
module purge

ml modenv/hiera 
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml CUDA/11.7.0
ml cuDNN/8.4.1.50-CUDA-11.7.0
ml NCCL/2.12.12-CUDA-11.7.0
ml Python/3.9.6-bare

set -euxo pipefail

SETUPVENV=true

VENV="./py396_gcc113_cu117"
if $SETUPVENV; then  
  python -m venv ${VENV}
fi

source "${VENV}/bin/activate"
  
if $SETUPVENV; then  
  pip install pip --upgrade
  pip install wheel
  pip install typing_extensions
  pip install -r requirements.yaml #install environment from requirement file
fi

           
ml Ninja
ml CMake/3.23.1

TORCH_VER="v1.12.1"
TORCH_DIR="./pytorch/"
git clone https://github.com/pytorch/pytorch.git ${TORCH_DIR}

## Have a git clone of PyTorch repo
cd ${TORCH_DIR}
## Update existing checkout
git checkout tags/${TORCH_VER}
git submodule sync
git submodule update --init --recursive 

python setup.py install 

echo Finished install of torch

## Install torch dependent packages
#pip install torch-scatter       # installed after build from source
#pip install torch-sparse        # installed after build from source
#pip install torch-geometric     # installed after build from source

echo Finished install of PyG
