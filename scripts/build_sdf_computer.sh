#/bin/bash

git submodule update
ROOT_DIR=$PWD

echo "======================================"
echo "Start to build tiny-cuda-nn"
echo "======================================"
cd $ROOT_DIR/thirdparty/tiny-cuda-nn && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j20 && cd bindings/torch && python setup.py install

echo "======================================"
echo "Start to build sdf_computer"
echo "======================================"
cd $ROOT_DIR/thirdparty/sdf_computer && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j20

echo "======================================"
echo "Start to build faiss"
echo "======================================"
cd $ROOT_DIR/thirdparty/faiss && cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAK_TOOLCHAIN_FILE=$vcpkg && cmake --build build --target faiss --config Release -j20 && cmake --build build --target swigfaiss --config Release -j20 && cd build/faiss/python && python setup.py install

echo "======================================"
echo "Start to build mesh2sdf"
echo "======================================"
cd $ROOT_DIR/thirdparty/mesh2sdf_cuda && python setup.py install
