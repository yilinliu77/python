import torch

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
    )

autoencoder = MeshAutoencoder(num_discrete_coors=128, pad_id=-1)
# checkpoint_autoencoder = torch.load('/mnt/d/meshgpt/autoencoder/last.ckpt')["state_dict"]
# checkpoint_autoencoder_ = {k[12:]: v for k, v in checkpoint_autoencoder.items() if 'autoencoder' in k}
# print(len(checkpoint_autoencoder.keys()), len(checkpoint_autoencoder_.keys()))
# autoencoder.load_state_dict(checkpoint_autoencoder_, strict=True)

transformer = MeshTransformer(autoencoder, max_seq_len=15144)

# autoencoder

autoencoder = MeshAutoencoder(
        num_discrete_coors=128
        )

# mock inputs

vertices = torch.randn((2, 121, 3))  # (batch, num vertices, coor (3))
faces = torch.randint(0, 121, (2, 64, 3))  # (batch, num faces, vertices (3))

# make sure faces are padded with `-1` for variable lengthed meshes

# forward in the faces
vertices = vertices.to('cuda')
faces = faces.to('cuda')
autoencoder.to('cuda')

loss = autoencoder(
        vertices=vertices,
        faces=faces
        )

loss.backward()

# after much training...
# you can pass in the raw face data above to train a transformer to model this sequence of face vertices

transformer = MeshTransformer(
        autoencoder,
        dim=512,
        max_seq_len=768,
        condition_on_text = True
        )

transformer.to('cuda')

loss = transformer(
        vertices=vertices,
        faces=faces,
        texts = ['a high chair', 'a small teapot'],
        )

loss.backward()

# after much training of transformer, you can now sample novel 3d assets

faces_coordinates, face_mask = transformer.generate(texts = ['a long table'])

# (batch, num faces, vertices (3), coordinates (3)), (batch, num faces)
# now post process for the generated 3d asset
