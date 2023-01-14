#/bin/bash

git submodule update
ROOT_DIR=$PWD

echo "Start to build tiny-cuda-nn"
cd $ROOT_DIR/thirdparty/tiny-cuda-nn && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j20 && cd bindings/torch && python setup.py install

echo "Start to build sdf_computer"
cd $ROOT_DIR/thirdparty/sdf_computer && cmake . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j20

echo "Start to build faiss"
cd $ROOT_DIR/thirdparty/faiss && cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAK_TOOLCHAIN_FILE=$vcpkg && cmake --build build --target faiss --config=Release -j20 && cmake . -B build --target swigfaiss --config=Release -j20 && cd build/faiss/python && python setup.py install
