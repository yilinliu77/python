conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::pythonocc-core==7.8.1
pip install diffusers vector-quantize-pytorch hydra-core tensorboard plyfile einops h5py opencv-python matplotlib scikit-image trimesh open3d pytorch-lightning torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
