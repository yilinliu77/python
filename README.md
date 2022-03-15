# visualDet3D

Some modifies to this package. Apply it and run `python setup.py build_ext --inplace` in both `thirdparty/visualDet3D/visualDet3D/networks/lib/ops/iou3d` and `thirdparty/visualDet3D/visualDet3D/networks/lib/ops/dcn`

- thirdparty/visualDet3D/visualDet3D/networks/lib/ops/iou3d/src/iou3d_kernel.cpp
    - 82: long * keep_data = keep.data_ptr<long>(); -> int64_t * keep_data = keep.data_ptr<int64_t>();
    - 100: unsigned long long remv_cpu[col_blocks]; -> unsigned long long* remv_cpu = new unsigned long long[col_blocks];
    - 132: long * keep_data = keep.data_ptr<long>(); -> int64_t * keep_data = keep.data_ptr<int64_t>();
    - 150: unsigned long long remv_cpu[col_blocks]; -> unsigned long long* remv_cpu = new unsigned long long[col_blocks];
    
- thirdparty/visualDet3D/visualDet3D/networks/lib/ops/iou3d/src/iou3d_kernel.cu
    - 81+: const float EPS = 1e-8;
    - 221+: const float EPS = 1e-8;
    - 304+: const float EPS = 1e-8;
  
- C:/repo/python/thirdparty/visualDet3D/visualDet3D/utils/utils.py
  - 139+: `sys.path.insert(0, "config")
    cfg = getattr(importlib.import_module(os.path.splitext(cfg_filename)[0].split("/")[1]), 'cfg')`
  - Comment the rest

- Installation
  pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  pip3 install open3d tqdm opencv-python hydra-core scikit-image matplotlib plyfile pytorch_lightning argparse omegaconf --user