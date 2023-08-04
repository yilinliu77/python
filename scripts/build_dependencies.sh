#!/bin/bash

read -p "Windows? (Y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[yY]$ ]]
then
    with_windows=true
else
    with_windows=false
fi
read -p "CUDA_ARCHITECTURES(86)? (Y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[yY]$ ]]
then
    CUDA_ARCHITECTURES=86
else
    CUDA_ARCHITECTURES=""
fi

git submodule update

multi_thread="-j20"

ROOT_DIR=$PWD

# Miniconda3
#https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

conda install -c intel mkl mkl-devel mkl-static mkl-include -y
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y
conda install -c pyg pyg -y
conda install -c conda-forge faiss-gpu -y
pip install numba PyMCubes pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python opencv-contrib-python ternausnet inplace_abn einops open3d ray[default]
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

echo "======================================"
echo "Start to build tiny-cuda-nn"
echo "======================================"
cd thirdparty/tiny-cuda-nn && export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release ${multi_thread} && cd bindings/torch && python setup.py install && cd ../../../../

echo "======================================"
echo "Start to build sdf_computer"
echo "======================================"
cd thirdparty/sdf_computer && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release ${multi_thread}  && cd ../../

# Not used
#echo "======================================"
#echo "Start to build faiss"
#echo "======================================"
#cd thirdparty/faiss && cmake -B build . -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$vcpkg && cmake --build build --target faiss --config Release ${multi_thread} && cmake --build build --target swigfaiss --config Release ${multi_thread} && cd build/faiss/python && python setup.py install  && cd ../../

# echo "======================================"
# echo "Start to build mesh2sdf"
# echo "======================================"
# cd thirdparty/mesh2sdf_cuda && python setup.py install
