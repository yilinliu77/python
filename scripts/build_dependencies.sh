#!/bin/bash

git submodule update --remote

multi_thread="-j20"

ROOT_DIR=$PWD

conda install -c conda-forge -c intel -c pyg -c pytorch -c nvidia mkl mkl-devel mkl-static mkl-include pyg faiss-gpu pytorch torchvision torchaudio pytorch-cuda=11.8
pip install numba PyMCubes pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python opencv-contrib-python ternausnet inplace_abn einops open3d ray[default]
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

echo "======================================"
echo "Start to build tiny-cuda-nn"
echo "======================================"
cd thirdparty/tiny-cuda-nn && export TCNN_CUDA_ARCHITECTURES=86 && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release ${multi_thread} && cd bindings/torch && python setup.py install && cd ../../../../

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
