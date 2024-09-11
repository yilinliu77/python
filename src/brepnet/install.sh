conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::pythonocc-core==7.8.1
pip install einops h5py opencv-python matplotlib scikit-image trimesh open3d pytorch-lightning torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
