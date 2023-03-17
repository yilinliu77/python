#!/bin/zsh

multi_thread="-j20"

ROOT_DIR=$PWD

pip install numba PyMCubes pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python opencv-contrib-python ternausnet inplace_abn einops open3d

echo "======================================"
echo "Start to build tiny-cuda-nn"
echo "======================================"
cd $ROOT_DIR/thirdparty/tiny-cuda-nn && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release ${multi_thread} && cd bindings/torch && python setup.py install

echo "======================================"
echo "Start to build sdf_computer"
echo "======================================"
cd $ROOT_DIR/thirdparty/sdf_computer && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release ${multi_thread}

echo "======================================"
echo "Start to build faiss"
echo "======================================"
cd $ROOT_DIR/thirdparty/faiss && cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$vcpkg && cmake --build build --target faiss --config Release ${multi_thread} && cmake --build build --target swigfaiss --config Release ${multi_thread} && cd build/faiss/python && python setup.py install

# Not used
# echo "======================================"
# echo "Start to build mesh2sdf"
# echo "======================================"
# cd $ROOT_DIR/thirdparty/mesh2sdf_cuda && python setup.py install
