import torch
import os
import numpy as np
from train import export_recon_faces
from mesh_render import combind_mesh, combind_mesh_with_rows, write_mesh_output
from einops import rearrange
import shutil
import open3d as o3d

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
    )



def read_data_and_pool(dataset_path, data_folders):
    pad_id = -1
    max_vertices_num = 0
    max_faces_num = 0
    folders = []
    vertices_tensor_list = []
    faces_tensor_list = []
    img_embed_list = []
    for folder in data_folders:
        folder_path = os.path.join(dataset_path, folder)
        tri_mesh_path = os.path.join(folder_path, "triangulation.ply")
        
        tri_mesh = o3d.io.read_triangle_mesh(tri_mesh_path)

        vertices = np.array(tri_mesh.vertices)
        triangle_vertices_idx = np.array(tri_mesh.triangles)
        triangle = vertices[triangle_vertices_idx]
        
        if triangle.shape[0] > 100:
            continue

        vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(device)
        faces_tensor = torch.tensor(triangle_vertices_idx, dtype=torch.long).to(device)
        vertices_tensor_list.append(vertices_tensor)
        faces_tensor_list.append(faces_tensor)
        max_vertices_num = vertices.shape[0] if vertices.shape[0] > max_vertices_num else max_vertices_num
        max_faces_num = triangle_vertices_idx.shape[0] if triangle_vertices_idx.shape[0] > max_faces_num \
            else max_faces_num
            
        img_embed = np.load(os.path.join(folder_path, "train_embed_vb16.npy"))
        img_embed = torch.tensor(img_embed, dtype=torch.float32).to(device)
        img_embed = rearrange(img_embed, 'img_num patch_num embed_dim -> (img_num patch_num) embed_dim')
        
        img_embed_list.append(img_embed)
        folders.append(folder)

    # pool
    for i in range(len(vertices_tensor_list)):
        vertices = vertices_tensor_list[i]
        faces = faces_tensor_list[i]
        vertices_num = vertices.shape[0]
        faces_num = faces.shape[0]
        if vertices_num < max_vertices_num:
            vertices_tensor_list[i] = torch.cat(
                    [vertices, torch.zeros(max_vertices_num - vertices_num, 3).to(device)])
        if faces_num < max_faces_num:
            faces_tensor_list[i] = torch.cat(
                    [faces, pad_id * torch.ones(max_faces_num - faces_num, 3).to(torch.long).to(device)])

    return torch.stack(vertices_tensor_list), torch.stack(faces_tensor_list), torch.stack(img_embed_list), folders


autoencoder = MeshAutoencoder(num_discrete_coors=128, pad_id=-1)
checkpoint_autoencoder = torch.load('/root/workspace/python/src/img2brep/meshgpt/tb_logs/autoencoder/version_1/checkpoints/epoch=566-step=2835.ckpt')["state_dict"]
checkpoint_autoencoder_ = {k[12:]: v for k, v in checkpoint_autoencoder.items() if 'autoencoder' in k}
autoencoder.load_state_dict(checkpoint_autoencoder_, strict=True)

is_condition_on_text = True
transformer = MeshTransformer(autoencoder, max_seq_len=768, 
                              coarse_pre_gateloop_depth = 2, 
                              fine_pre_gateloop_depth= 2,
                              condition_on_text=is_condition_on_text)
if is_condition_on_text:
    checkpoint_transformer = torch.load('/root/workspace/python/src/img2brep/meshgpt/tb_logs/transformer/version_5/checkpoints/epoch=3599-step=21600.ckpt')["state_dict"]
else:
    checkpoint_transformer = torch.load('/root/workspace/python/src/img2brep/meshgpt/tb_logs/transformer/version_5/checkpoints/epoch=3599-step=21600.ckpt')["state_dict"]
    
checkpoint_transformer_ = {}
for k, v in checkpoint_transformer.items():
    if 'transformer.' in k:
        checkpoint_transformer_[k[12:]] = v
    elif 'model.' in k:
        checkpoint_transformer_[k[6:]] = v

transformer.load_state_dict(checkpoint_transformer_, strict=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
autoencoder.to(device)
transformer.to(device)


os.makedirs('output', exist_ok=True)
os.makedirs('output_with_img_condition', exist_ok=True)

dataset_path = "/mnt/d/meshgpt/0planar_shapes_100"
data_folders = os.listdir(dataset_path)
data_folders.sort()

def gen_test():
    folder = "00000325"
    img_embed = np.load(os.path.join(dataset_path, folder, "train_embed.npy"))
    img_embed = torch.tensor(img_embed, dtype=torch.float32).to('cuda')
    img_embed = rearrange(img_embed, 'img_num patch_num embed_dim -> (img_num patch_num) embed_dim')
    img_embed = img_embed.unsqueeze(0)
    print(img_embed.shape)
    faces_coordinates, face_mask = transformer.generate(text_embeds=img_embed)
    print(faces_coordinates.shape, face_mask.shape)
    triangles = faces_coordinates[face_mask]
    print(triangles.shape)
    export_recon_faces(triangles, f'output_with_img_condition/generated_mesh.ply')

def uncondititon_gen(gen_times = 5, batch_size = 2, ):
    for k in range(gen_times):
        faces_coordinates, face_mask = transformer.generate(batch_size=batch_size, text_embeds=img_embed.tile(batch_size, 1, 1))
        print(faces_coordinates.shape, face_mask.shape)

        for i in range(faces_coordinates.shape[0]):
            triangles = faces_coordinates[i][face_mask[i]]
            export_recon_faces(triangles, f'output_without_img_condition/generated_mesh{(batch_size*k+i):05}.ply')
            
def condition_gen(img_embeds):
    print("img_embeds shape:",img_embeds.shape)
    for i in range(img_embeds.shape[0]):
        img_embed = img_embeds[i].unsqueeze(0)
        print(img_embed.shape)
        faces_coordinates, face_mask = transformer.generate(text_embeds=img_embed)
        print(faces_coordinates.shape, face_mask.shape)
        triangles = faces_coordinates[face_mask]
        print(triangles.shape)
        export_recon_faces(triangles, f'output_with_img_condition/generated_mesh{i:05}.ply')
        
def condition_gen_one(img_embed, output_path):
    img_embed = img_embed.unsqueeze(0) if img_embed.dim() == 2 else img_embed
    # print(img_embed.shape)
    # faces_coordinates, face_mask = transformer.generate(text_embeds=img_embed, temperature=0)
    faces_coordinates, face_mask = transformer.generate(text_embeds=img_embed, temperature=0.0, cond_scale=1)
    
    # print(faces_coordinates.shape, face_mask.shape)
    triangles = faces_coordinates[face_mask]
    # print(triangles.shape)
    export_recon_faces(triangles, output_path)


output_path = 'output_with_img_condition'

# folders = ["00000325","00000797"]

vertices_all, faces_all, img_embed_all, folders = read_data_and_pool(dataset_path, data_folders = data_folders)

triangles_list = []
src_triangles_list = []
for i in range(len(folders)):
    folder = folders[i]
    print(folder)
    vertices, faces, img_embed = vertices_all[i], faces_all[i], img_embed_all[i]
    img_embed = img_embed.unsqueeze(0) if img_embed.dim() == 2 else img_embed
    src_triangles_list.append(vertices[faces])
    
    tokens = autoencoder.tokenize(vertices=vertices, faces=faces)
    num_tokens = int(tokens.shape[0] * 0.1) 
    num_tokens = 1
    
    prompt = tokens.flatten()[:num_tokens].unsqueeze(0)
    prompt=None
    faces_coordinates, face_mask = transformer.generate(text_embeds=img_embed, cache_kv=False, prompt=prompt,temperature=1, cond_scale=1)
    triangles = faces_coordinates[face_mask]
    triangles_list.append(triangles.clone())
    
    export_recon_faces(triangles, os.path.join(output_path, f'{folder}_gen.ply'))
    shutil.copy(os.path.join(dataset_path, folder, "triangulation.ply"), os.path.join(output_path, f'{folder}_src.ply'))

rows_num = 10
triangles_list = [triangles_list[i:min(i+rows_num, len(triangles_list))] for i in range(0, len(triangles_list), rows_num)]
src_triangles_list = [src_triangles_list[i:min(i+rows_num, len(src_triangles_list))] for i in range(0, len(src_triangles_list), rows_num)]

combind_mesh_with_rows(os.path.join(output_path, f'all_gen.obj'), triangles_list)
combind_mesh_with_rows(os.path.join(output_path, f'all_src.obj'), src_triangles_list)
     
# out_path = 'output_with_img_condition'
# folders = ["00000325","00000797"]
# for folder in folders:
#     print(folder)
#     img_embed = np.load(os.path.join(dataset_path, folder, "train_embed_vb16.npy"))
#     img_embed = torch.tensor(img_embed, dtype=torch.float32).to(device)
#     img_embed = rearrange(img_embed, 'img_num patch_num embed_dim -> (img_num patch_num) embed_dim')
#     img_embed = img_embed[0]
#     condition_gen_one(img_embed, os.path.join(out_path, f'{folder}_gen.ply'))
#     shutil.copy(os.path.join(dataset_path, folder, "triangulation.ply"), os.path.join(out_path, f'{folder}_src.ply'))