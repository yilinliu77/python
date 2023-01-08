import torch # pytorch backend
import torchvision # CV models
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
pygm.BACKEND = 'pytorch' # set default backend for pygmtools

if __name__ == '__main__':
    obj_resize = (256, 256)
    data_root = r"D:/DATASET/WILLOW-ObjectClass_dataset/WILLOW-ObjectClass/Duck/"
    img1 = Image.open(data_root + '060_0000.png')
    img2 = Image.open(data_root + '060_0002.png')
    kpts1 = torch.tensor(sio.loadmat(data_root + '060_0000.mat')['pts_coord'])
    kpts2 = torch.tensor(sio.loadmat(data_root + '060_0002.mat')['pts_coord'])
    kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
    kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
    kpts2[0] = kpts2[0] * obj_resize[0] / img2.size[0]
    kpts2[1] = kpts2[1] * obj_resize[1] / img2.size[1]
    img1 = img1.resize(obj_resize, resample=Image.BILINEAR)
    img2 = img2.resize(obj_resize, resample=Image.BILINEAR)
    torch_img1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(
        0)  # shape: BxCxHxW
    torch_img2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(
        0)  # shape: BxCxHxW

    def plot_image_with_graph(img, kpt, A=None):
        plt.imshow(img)
        plt.scatter(kpt[0], kpt[1], c='w', edgecolors='k')
        if A is not None:
            for idx in torch.nonzero(A, as_tuple=False):
                plt.plot((kpt[0, idx[0]], kpt[0, idx[1]]), (kpt[1, idx[0]], kpt[1, idx[1]]), 'k-')


    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Image 1')
    plot_image_with_graph(img1, kpts1)
    plt.subplot(1, 2, 2)
    plt.title('Image 2')
    plot_image_with_graph(img2, kpts2)

    plt.show()

    def delaunay_triangulation(kpt):
        d = spa.Delaunay(kpt.numpy().transpose())
        A = torch.zeros(len(kpt[0]), len(kpt[0]))
        for simplex in d.simplices:
            for pair in itertools.permutations(simplex, 2):
                A[pair] = 1
        return A

    A1 = delaunay_triangulation(kpts1)
    A2 = delaunay_triangulation(kpts2)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Image 1 with Graphs')
    plot_image_with_graph(img1, kpts1, A1)
    plt.subplot(1, 2, 2)
    plt.title('Image 2 with Graphs')
    plot_image_with_graph(img2, kpts2, A2)

    plt.show()

    vgg16_cnn = torchvision.models.vgg16_bn(True)

    class CNNNet(torch.nn.Module):
        def __init__(self, vgg16_module):
            super(CNNNet, self).__init__()
            # The naming of the layers follow ThinkMatch convention to load pretrained models.
            self.node_layers = torch.nn.Sequential(*[_ for _ in vgg16_module.features[:31]])
            self.edge_layers = torch.nn.Sequential(*[_ for _ in vgg16_module.features[31:38]])

        def forward(self, inp_img):
            feat_local = self.node_layers(inp_img)
            feat_global = self.edge_layers(feat_local)
            return feat_local, feat_global

    cnn = CNNNet(vgg16_cnn)
    path = pygm.utils.download('vgg16_pca_voc_pytorch.pt',
                               'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1JnX3cSPvRYBSrDKVwByzp7CADgVCJCO_')
    if torch.cuda.is_available():
        map_location = torch.device('cuda:0')
    else:
        map_location = torch.device('cpu')
    cnn.load_state_dict(torch.load(path, map_location=map_location), strict=False)
    with torch.set_grad_enabled(False):
        feat1_local, feat1_global = cnn(torch_img1)
        feat2_local, feat2_global = cnn(torch_img2)


    def l2norm(node_feat):
        return torch.nn.functional.local_response_norm(
            node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)


    feat1_local = l2norm(feat1_local)
    feat1_global = l2norm(feat1_global)
    feat2_local = l2norm(feat2_local)
    feat2_global = l2norm(feat2_global)

    feat1_local_upsample = torch.nn.functional.interpolate(feat1_local, obj_resize, mode='bilinear')
    feat1_global_upsample = torch.nn.functional.interpolate(feat1_global, obj_resize, mode='bilinear')
    feat2_local_upsample = torch.nn.functional.interpolate(feat2_local, obj_resize, mode='bilinear')
    feat2_global_upsample = torch.nn.functional.interpolate(feat2_global, obj_resize, mode='bilinear')
    feat1_upsample = torch.cat((feat1_local_upsample, feat1_global_upsample), dim=1)
    feat2_upsample = torch.cat((feat2_local_upsample, feat2_global_upsample), dim=1)
    num_features = feat1_upsample.shape[1]


    pca_dim_reduc = PCAdimReduc(n_components=3, whiten=True)
    feat_dim_reduc = pca_dim_reduc.fit_transform(
        np.concatenate((
            feat1_upsample.permute(0, 2, 3, 1).reshape(-1, num_features).numpy(),
            feat2_upsample.permute(0, 2, 3, 1).reshape(-1, num_features).numpy()
        ), axis=0)
    )
    feat_dim_reduc = feat_dim_reduc / np.max(np.abs(feat_dim_reduc), axis=0, keepdims=True) / 2 + 0.5
    feat1_dim_reduc = feat_dim_reduc[:obj_resize[0] * obj_resize[1], :]
    feat2_dim_reduc = feat_dim_reduc[obj_resize[0] * obj_resize[1]:, :]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Image 1 with CNN features')
    plot_image_with_graph(img1, kpts1, A1)
    plt.imshow(feat1_dim_reduc.reshape(obj_resize[0], obj_resize[1], 3), alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.title('Image 2 with CNN features')
    plot_image_with_graph(img2, kpts2, A2)
    plt.imshow(feat2_dim_reduc.reshape(obj_resize[0], obj_resize[1], 3), alpha=0.5)

    plt.show()

    rounded_kpts1 = torch.round(kpts1).to(dtype=torch.long)
    rounded_kpts2 = torch.round(kpts2).to(dtype=torch.long)
    node1 = feat1_upsample[0, :, rounded_kpts1[0], rounded_kpts1[1]].t()  # shape: NxC
    node2 = feat2_upsample[0, :, rounded_kpts2[0], rounded_kpts2[1]].t()  # shape: NxC

    conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    conn2, edge2 = pygm.utils.dense_to_sparse(A2)
    import functools
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1)  # set affinity function
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, edge_aff_fn=gaussian_aff)

    X = pygm.rrwm(K, kpts1.shape[1], kpts2.shape[1])
    # X = pygm.pca_gm(node1, node2, A1, A2, pretrain='voc')
    X = pygm.hungarian(X)

    plt.figure(figsize=(8, 4))
    plt.suptitle('Image Matching Result by RRWM')
    ax1 = plt.subplot(1, 2, 1)
    plot_image_with_graph(img1, kpts1, A1)
    ax2 = plt.subplot(1, 2, 2)
    plot_image_with_graph(img2, kpts2, A2)
    for i in range(X.shape[0]):
        j = torch.argmax(X[i]).item()
        con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color="red" if i != j else "green")
        plt.gca().add_artist(con)

    plt.show()