conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::pythonocc-core==7.8.1
pip install wandb point_cloud_utils chamferdist diffusers vector-quantize-pytorch hydra-core tensorboard ray plyfile einops h5py opencv-python matplotlib scikit-image trimesh open3d pytorch-lightning torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
apt install libgl-dev
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'

# LFD
apt install freeglut3 freeglut3-dev libosmesa6-dev libglu1-mesa-dev libglu1-mesa xserver-xorg-video-dummy