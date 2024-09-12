import os

import torch
import numpy as np

import multiprocessing
from tqdm import tqdm
import trimesh

# import pandas as pd

from chamferdist import ChamferDistance

import random

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.DataExchange import read_step_file, write_step_file, write_stl_file

import traceback
import time, os


def write_ply(points, path):
    point_cloud = trimesh.PointCloud(points)
    point_cloud.export(path)


def create_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def is_vertex_close(p1, p2, tol):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def get_edge_length(edge, NUM_SEGMENTS=100):
    curve_data = BRep_Tool.Curve(edge)
    if curve_data and len(curve_data) == 3:
        curve_handle, first, last = curve_data
        segment_length = (last - first) / NUM_SEGMENTS
        edge_length = 0
        for i in range(NUM_SEGMENTS):
            u1 = first + segment_length * i
            u2 = first + segment_length * (i + 1)
            edge_length += np.linalg.norm(np.array(curve_handle.Value(u1).Coord()) - np.array(curve_handle.Value(u2).Coord()))
        return edge_length


def read_step_and_get_data(step_file_path, NUM_SAMPLE_EDGE_UNIT=100):
    if not os.path.exists(step_file_path):
        return None, None

    shape = read_step_file(step_file_path, verbosity=False)

    # face
    # face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # faces = []
    # while face_explorer.More():
    #     face = face_explorer.Current()
    #     faces.append(face)
    #     face_explorer.Next()
    # gen_faces = np.stack(faces)

    # vertex
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        point = BRep_Tool.Pnt(vertex)
        # check if the point is close to the previous point
        is_close = False
        for v in vertexes:
            if is_vertex_close(v, (point.X(), point.Y(), point.Z()), 1e-4):
                is_close = True
                break
        if not is_close:
            vertexes.append((point.X(), point.Y(), point.Z()))
        vertex_explorer.Next()

    # edge
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    topods_edge_list = []
    edge_points = []
    while edge_explorer.More():
        edge = edge_explorer.Current()
        is_saved = False
        for each in topods_edge_list:
            if each.IsEqual(edge) or each.IsEqual(edge.Reversed()):
                is_saved = True
                break
        if not is_saved:
            topods_edge_list.append(edge)
            curve_data = BRep_Tool.Curve(edge)
            if curve_data and len(curve_data) == 3:
                curve_handle, first, last = curve_data
                NUM_SAMPLE_EDGE = int(get_edge_length(edge, NUM_SEGMENTS=32) * NUM_SAMPLE_EDGE_UNIT)
                u_values = np.linspace(first, last, NUM_SAMPLE_EDGE)
                points_on_edge = []
                for u in u_values:
                    point = curve_handle.Value(u)
                    points_on_edge.append((point.X(), point.Y(), point.Z()))
                if len(points_on_edge) != 0:
                    points_on_edge = np.array(points_on_edge)
                    points_on_edge.reshape(-1, 3)
                    edge_points.append(points_on_edge)
        edge_explorer.Next()
    return vertexes, edge_points


class SamplePointsAndComputeCD:
    """
    Perform sampleing of points.
    """

    def __init__(self, gt_root, root_path, SAMPLE_NUM=100000, visable_gpu_id=[0, 1, 2, 3, 4, 5, 6, 7],
                 num_cpus=32, batch_size=32, is_save_pc=True, is_debug=False):
        self.gt_root = gt_root
        self.root_path = root_path
        self.folder_names = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
        # self.folder_names = np.load(r"/mnt/d/failed_face_folder.npz")['arr_0']
        self.folder_names.sort()
        if is_debug:
            self.folder_names = self.folder_names[:32]
        self.gen_name_condicate = ['recon_brep.stl', 'recon_brep_invaild.stl', 'recon_brep_compound.stl']
        self.SAMPLE_NUM = SAMPLE_NUM
        self.SAMPLE_NUM_FACE = int(10000)

        self.return_dict = {}
        self.exception_folders = []

        self.is_save_pc = is_save_pc
        self.chamfer_distance = ChamferDistance()
        self.num_cpus = num_cpus
        self.batch_size = batch_size
        self.gpu_list = [torch.device(f"cuda:{i}") for i in visable_gpu_id]
        print("Using GPU list: ", self.gpu_list)

        self.return_dict_save_path = os.path.join(os.path.dirname(self.root_path), 'return_dict.npz')

    def exception_handle(self):
        print("Exception folders: ", self.exception_folders)

    def can_pass(self, folder_name):
        if os.path.exists(os.path.join(self.root_path, folder_name, 'eval.npz')):
            return True
        else:
            return False

    def process_one(self, folder_name):
        if os.path.exists(os.path.join(self.root_path, folder_name, 'error.txt')):
            os.remove(os.path.join(self.root_path, folder_name, 'error.txt'))
        if os.path.exists(os.path.join(self.root_path, folder_name, 'eval.npz')):
            os.remove(os.path.join(self.root_path, folder_name, 'eval.npz'))

        result = {
            'num_recon_face'  : 0,
            'num_gt_face'     : 0,
            'face_acc_cd'     : [],
            'face_com_cd'     : [],
            'face_cd'         : [],

            'num_recon_edge'  : 0,
            'num_gt_edge'     : 0,
            'edge_acc_cd'     : [],
            'edge_com_cd'     : [],
            'edge_cd'         : [],

            'num_recon_vertex': 0,
            'num_gt_vertex'   : 0,
            'vertex_acc_cd'   : [],
            'vertex_com_cd'   : [],
            'vertex_cd'       : [],

            'stl_type'        : '',
            'stl_acc_cd'      : 0,
            'stl_com_cd'      : 0,
            'stl_cd'          : 0,
        }
        device = random.choice(self.gpu_list)
        chamfer_distance = ChamferDistance()

        # gt info
        gt_mesh_path = os.path.join(self.gt_root, folder_name, 'mesh.ply')
        gt_mesh = trimesh.load(gt_mesh_path)
        gt_pc, _ = trimesh.sample.sample_surface(gt_mesh, self.SAMPLE_NUM)
        gt_pc_tensor = torch.from_numpy(gt_pc).float().to(device)

        data_npz = np.load(os.path.join(self.gt_root, folder_name, 'data.npz'))
        result['num_gt_face'] = data_npz['sample_points_faces'].shape[0]
        # result['num_gt_edge'] = data_npz['sample_points_lines'].shape[0]
        # result['num_gt_vertex'] = data_npz['sample_points_vertices'].shape[0]

        gt_step_path = os.path.join(self.gt_root, folder_name, 'normalized_shape.step')
        gt_vertexes, gt_edge_points = read_step_and_get_data(gt_step_path)
        result['num_gt_edge'] = len(gt_edge_points)
        result['num_gt_vertex'] = len(gt_vertexes)
        gt_vertexes = np.stack(gt_vertexes, axis=0)
        gt_edge_points = np.concatenate(gt_edge_points, axis=0)
        gt_vertex_tensor = torch.from_numpy(gt_vertexes).float().to(device)
        gt_edge_tensor = torch.from_numpy(gt_edge_points).float().to(device)

        # face
        recon_face_dir = os.path.join(self.root_path, folder_name, 'recon_face')
        recon_stl_name = [f for f in os.listdir(recon_face_dir) if f.endswith('.stl')]
        if os.path.exists(recon_face_dir) and len(os.listdir(recon_face_dir)) != 0:
            result['num_recon_face'] = len(recon_stl_name)
            recon_face_stl_mesh = trimesh.util.concatenate([trimesh.load(os.path.join(recon_face_dir, f)) for f in recon_stl_name])
            recon_face_pc, _ = trimesh.sample.sample_surface(recon_face_stl_mesh, self.SAMPLE_NUM)
            recon_face_pc_tensor = torch.from_numpy(recon_face_pc).float().to(device)
            acc_cd = chamfer_distance(recon_face_pc_tensor.unsqueeze(0), gt_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            com_cd = chamfer_distance(gt_pc_tensor.unsqueeze(0), recon_face_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            cd = (acc_cd + com_cd) / 2
            result['face_acc_cd'].append(acc_cd)
            result['face_com_cd'].append(com_cd)
            result['face_cd'].append(cd)

        # read the edge and vertex from gen step file
        if os.path.exists(os.path.join(self.root_path, folder_name, 'recon_brep.step')):
            step_file_path = os.path.join(self.root_path, folder_name, 'recon_brep.step')
            gen_vertexes, gen_edge_points = read_step_and_get_data(step_file_path)
        else:
            gen_vertexes, gen_edge_points = None, None

        # edge
        if gen_edge_points is not None:
            result['num_recon_edge'] = len(gen_edge_points)
            gen_edge_points = np.concatenate(gen_edge_points, axis=0)
            recon_edge_pc_tensor = torch.from_numpy(gen_edge_points).float().to(device)
            acc_cd = chamfer_distance(recon_edge_pc_tensor.unsqueeze(0), gt_edge_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            com_cd = chamfer_distance(gt_edge_tensor.unsqueeze(0), recon_edge_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            cd = (acc_cd + com_cd) / 2
            result['edge_acc_cd'].append(acc_cd)
            result['edge_com_cd'].append(com_cd)
            result['edge_cd'].append(cd)

        # vertex
        if gen_vertexes is not None:
            result['num_recon_vertex'] = len(gen_vertexes)
            gen_vertexes = np.stack(gen_vertexes, axis=0)
            recon_vertex_pc_tensor = torch.from_numpy(gen_vertexes).float().to(device)
            acc_cd = chamfer_distance(recon_vertex_pc_tensor.unsqueeze(0), gt_vertex_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            com_cd = chamfer_distance(gt_vertex_tensor.unsqueeze(0), recon_vertex_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            cd = (acc_cd + com_cd) / 2
            result['vertex_acc_cd'].append(acc_cd)
            result['vertex_com_cd'].append(com_cd)
            result['vertex_cd'].append(cd)

        # shell or soild
        gen_name = None
        for condicate in self.gen_name_condicate:
            if os.path.exists(os.path.join(self.root_path, folder_name, condicate)):
                gen_name = condicate
                break

        if gen_name is not None:
            result['stl_type'] = gen_name
            gen_stl_path = os.path.join(self.root_path, folder_name, gen_name)
            gen_mesh = trimesh.load(gen_stl_path)
        else:
            if os.path.exists(recon_face_dir):
                gen_mesh = trimesh.util.concatenate([trimesh.load(os.path.join(recon_face_dir, f))
                                                     for f in os.listdir(recon_face_dir) if f.endswith('.stl')])
            else:
                gen_mesh = None

        if gen_mesh is not None:
            gen_pc, _ = trimesh.sample.sample_surface(gen_mesh, self.SAMPLE_NUM)
            gen_pc_tensor = torch.from_numpy(gen_pc).float().to(device)
            acc_cd = chamfer_distance(gen_pc_tensor.unsqueeze(0), gt_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            com_cd = chamfer_distance(gt_pc_tensor.unsqueeze(0), gen_pc_tensor.unsqueeze(0),
                                      bidirectional=False, point_reduction='mean').cpu().item()
            result['stl_acc_cd'] = acc_cd
            result['stl_com_cd'] = com_cd
            result['stl_cd'] = (acc_cd + com_cd) / 2

        np.savez_compressed(os.path.join(self.root_path, folder_name, 'eval.npz'), result=result, allow_pickle=True)

    def process_list(self, folder_name_list):
        for folder_name in folder_name_list:
            if os.path.exists(os.path.join(self.root_path, folder_name, 'error.txt')):
                os.remove(os.path.join(self.root_path, folder_name, 'error.txt'))
            try:
                self.process_one(folder_name)
            except Exception as e:
                error_info = traceback.format_exc()
                with open(os.path.join(self.root_path, folder_name, 'error.txt'), 'w') as f:
                    f.write(f"Error in folder {folder_name}:\n{error_info}")
            # return_dict[folder_name] = result

    def run_sequntial(self):
        print("Running sequentially...")
        for folder_name in tqdm(self.folder_names):
            self.process_one(folder_name)

    def run_parallel(self):
        print("Running in parallel...")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        print("Number of CPUs: ", self.num_cpus)
        pool = multiprocessing.Pool(self.num_cpus)
        jobs = []

        print("Applying async task")
        for batch in create_batches(self.folder_names, self.batch_size):
            job = pool.apply_async(self.process_list, (batch,))
            jobs.append(job)

        print("Apply async task done")
        bar = tqdm(total=len(self.folder_names))
        while len(jobs) > 0:
            for job in jobs[:]:
                if job.ready():
                    bar.update(self.batch_size)
                    jobs.remove(job)
            time.sleep(1)

        # for job in tqdm(jobs):
        #     job.wait()
        pool.close()
        pool.join()

    def run(self, is_parallel=False, is_save=False, is_info=False):
        if is_parallel:
            self.run_parallel()
        else:
            self.run_sequntial()
        if is_save:
            self.save()
        if is_info:
            self.info()

    def save(self):
        print("Checking and Saving...")
        return_dict = {}
        for folder_name in tqdm(self.folder_names):
            if not os.path.exists(os.path.join(self.root_path, folder_name, 'eval.npz')):
                self.exception_folders.append(folder_name)
            else:
                result = np.load(os.path.join(self.root_path, folder_name, 'eval.npz'), allow_pickle=True)['result'].item()
                if result['num_recon_face'] == 0:
                    self.exception_folders.append(folder_name)
                return_dict[folder_name] = result

        np.savez_compressed(self.return_dict_save_path, return_dict=return_dict, allow_pickle=True)
        if len(self.exception_folders) != 0:
            np.savez_compressed(os.path.join(os.path.dirname(self.root_path), 'exception_folders.npz'),
                                exception_folders=self.exception_folders)
        print(f"Len exception folders: {len(self.exception_folders)}")
        print(f"Len return dict: {len(return_dict)}")
        print("Return dict is saved in {}".format(self.return_dict_save_path))

    def info(self):
        print("Loading return dict...")
        self.return_dict = np.load(self.return_dict_save_path, allow_pickle=True)['return_dict'].item()
        print(type(self.return_dict))
        print("Return dict length: ", len(self.return_dict.keys()))

        print("Computing statistics...")
        sum_recon_face, sum_gt_face = 0, 0
        sum_recon_edge, sum_gt_edge = 0, 0
        sum_recon_vertex, sum_gt_vertex = 0, 0

        all_face_acc_cd, all_face_com_cd, all_face_cd = [], [], []
        all_edge_acc_cd, all_edge_com_cd, all_edge_cd = [], [], []
        all_vertex_acc_cd, all_vertex_com_cd, all_vertex_cd = [], [], []

        all_stl_acc_cd, all_stl_com_cd, all_stl_cd = [], [], []
        num_soild, num_shell, num_compound = 0, 0, 0

        solid_acc_cd, solid_com_cd, solid_cd = [], [], []
        shell_acc_cd, shell_com_cd, shell_cd = [], [], []
        compound_acc_cd, compound_com_cd, compound_cd = [], [], []

        for each in tqdm(self.return_dict.values()):
            sum_recon_face += int(each['num_recon_face'])
            sum_gt_face += int(each['num_gt_face'])
            sum_recon_edge += int(each['num_recon_edge'])
            sum_gt_edge += int(each['num_gt_edge'])
            sum_recon_vertex += int(each['num_recon_vertex'])
            sum_gt_vertex += int(each['num_gt_vertex'])

            all_face_acc_cd.extend(each['face_acc_cd'])
            all_face_com_cd.extend(each['face_com_cd'])
            all_face_cd.extend(each['face_cd'])

            all_edge_acc_cd.extend(each['edge_acc_cd'])
            all_edge_com_cd.extend(each['edge_com_cd'])
            all_edge_cd.extend(each['edge_cd'])

            all_vertex_acc_cd.extend(each['vertex_acc_cd'])
            all_vertex_com_cd.extend(each['vertex_com_cd'])
            all_vertex_cd.extend(each['vertex_cd'])

            all_stl_acc_cd.append(each['stl_acc_cd'])
            all_stl_com_cd.append(each['stl_com_cd'])
            all_stl_cd.append(each['stl_cd'])

            if each['stl_type'] == 'recon_brep.stl':
                num_soild += 1
                solid_acc_cd.append(each['stl_acc_cd'])
                solid_com_cd.append(each['stl_com_cd'])
                solid_cd.append(each['stl_cd'])
            elif each['stl_type'] == 'recon_brep_compound.stl':
                num_shell += 1
                shell_acc_cd.append(each['stl_acc_cd'])
                shell_com_cd.append(each['stl_com_cd'])
                shell_cd.append(each['stl_cd'])
            elif each['stl_type'] == 'recon_brep_invaild.stl':
                num_compound += 1
                compound_acc_cd.append(each['stl_acc_cd'])
                compound_com_cd.append(each['stl_com_cd'])
                compound_cd.append(each['stl_cd'])
            else:
                pass

        # print the statistics
        print("\nFace")
        print("Recon: ", sum_recon_face)
        print("GT: ", sum_gt_face)
        print("Average ACC CD: ", np.mean(all_face_acc_cd))
        print("Average COM CD: ", np.mean(all_face_com_cd))
        print("Average CD: ", np.mean(all_face_cd))

        print("\nEdge")
        print("Recon: ", sum_recon_edge)
        print("GT: ", sum_gt_edge)
        print("Average ACC CD: ", np.mean(all_edge_acc_cd))
        print("Average COM CD: ", np.mean(all_edge_com_cd))
        print("Average CD: ", np.mean(all_edge_cd))

        print("\nVertex")
        print("Recon: ", sum_recon_vertex)
        print("GT: ", sum_gt_vertex)
        print("Average ACC CD: ", np.mean(all_vertex_acc_cd))
        print("Average COM CD: ", np.mean(all_vertex_com_cd))
        print("Average CD: ", np.mean(all_vertex_cd))

        print("\nSoild: ", num_soild)
        print("Average Acc CD: ", np.mean(solid_acc_cd))
        print("Average Com CD: ", np.mean(solid_com_cd))
        print("Average CD: ", np.mean(solid_cd))

        print("\nShell: ", num_shell)
        print("Average Acc CD: ", np.mean(shell_acc_cd))
        print("Average Com CD: ", np.mean(shell_com_cd))
        print("Average CD: ", np.mean(shell_cd))

        print("\nCompound: ", num_compound)
        print("Average Acc CD: ", np.mean(compound_acc_cd))
        print("Average Com CD: ", np.mean(compound_com_cd))
        print("Average CD: ", np.mean(compound_cd))

        # data = pd.DataFrame(all_stl_cd, columns=['all_stl_cd'])
        # print(data.info())
        # print(data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    def rerun_check(self):
        print("Checking CD")
        self.CD = np.load(os.path.join(os.path.dirname(self.root_path), 'cd.npz'))
        folder_name_list = []
        for folder_name, cd in tqdm(self.CD.items()):
            if cd > 1e-4:
                folder_name_list.append(folder_name)
        print(len(folder_name_list))
        # save the failed folder names
        np.savez(os.path.join(os.path.dirname(self.root_path), 'recheck.npz'), folder_name_list)
        print("Failed folder names are saved in {}".format(os.path.join(os.path.dirname(self.root_path), 'failed_folder_names.npy')))

    def find_complex(self):
        import shutil
        save_root = os.path.join(os.path.dirname(self.root_path), 'segment_by_face_num')
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root, exist_ok=False)
        seg_save_root = [os.path.join(save_root, 'face>30'), os.path.join(save_root, 'face>20'), os.path.join(save_root, 'face>10'),
                         os.path.join(save_root, 'else')]
        for each in seg_save_root:
            os.makedirs(each, exist_ok=True)

        random.shuffle(self.folder_names)
        self.folder_names = self.folder_names[0:2000]
        for folder_name in tqdm(self.folder_names):
            data_npz = np.load(os.path.join(self.root_path, folder_name, 'data.npz'))
            if data_npz['sample_points_faces'].shape[0] > 30:
                shutil.copytree(os.path.join(self.root_path, folder_name), os.path.join(seg_save_root[0], folder_name))
            elif data_npz['sample_points_faces'].shape[0] > 20:
                shutil.copytree(os.path.join(self.root_path, folder_name), os.path.join(seg_save_root[1], folder_name))
            elif data_npz['sample_points_faces'].shape[0] > 10:
                shutil.copytree(os.path.join(self.root_path, folder_name), os.path.join(seg_save_root[2], folder_name))
            else:
                continue
                shutil.copytree(os.path.join(self.root_path, folder_name), os.path.join(seg_save_root[3], folder_name))


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    app = SamplePointsAndComputeCD(gt_root=r"E:\data\img2brep\deepcad_whole_train_v5",
                                   root_path=r"E:\data\img2brep\deepcad_whole_train_v5_out",
                                   visable_gpu_id=[0], is_save_pc=False, is_debug=False)
    app.run(is_parallel=False, is_save=True, is_info=True)
    # app.info()
    # app.find_complex()
