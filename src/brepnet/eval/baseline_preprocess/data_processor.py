from __future__ import annotations

import trimesh
import json
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge, TopoDS_Face

from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt

from OCC.Core.gp import gp_Pnt

from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface, ShapeAnalysis_Curve, shapeanalysis
from typing import List

from shared.occ_utils import get_primitives, get_triangulations, get_points_along_edge, get_curve_length

class DataProcessor(ABC):
    _BSPLINE_CURVE_DEGREE_MIN = 0
    _BSPLINE_CURVE_DEGREE_MAX = 8
    _BSPLINE_FACE_DEGREE_MIN = 3
    _BSPLINE_FACE_DEGREE_MAX = 8
    _CONTINUITY = GeomAbs_C2
    _CURVELENGTH = 1
    _CURVATURE = 1
    _TORSION = 1
    _EDGE_FITTING_TOLERANCE = [1e-3, 5e-3, 8e-3, 5e-2]
    _FACE_FITTING_TOLERANCE = [1e-3, 1e-2, 3e-2, 5e-2, 8e-2]
    _U_AXIS = 0
    _V_AXIS = 1

    def __init__(self, file_path: str | Path):
        self._transfer_tolerance = 1e-6
        self.file_path = file_path
        self.file_content = self._read_file(self.file_path)
        self._vertex_sample_points = self._get_vertex_array()
        self._edge_sample_points = self._get_edge_array()
        self._face_sample_points = self._get_face_array()
        self._vertices = self.get_vertices()
        self._edges = self.get_edges()
        self._faces = self.get_faces(self.TransferTolerance)
        self._face_edge = self._get_face_edge()
        self._edge_vertex = self._get_edge_vertex()
        assert len(self.Vertices) == self.VertexPositions.shape[0]
        assert len(self.Edges) == self.EdgeSamplePoints.shape[0]
        assert len(self.Faces) == self.FaceSamplePoints.shape[0]

    def get_vertices(self) -> List[TopoDS_Vertex]:
        """
        Get the vertices in the BRep form

        Returns:
            List[TopoDS_Vertex]: List of vertices
        """
        vertices = []
        for vertex_position in self.VertexPositions:
            vertex_position = vertex_position[0]
            vertices.append(BRepBuilderAPI_MakeVertex(gp_Pnt(*vertex_position)))
        return vertices

    def get_edges(self) -> List[TopoDS_Edge]:
        """
        Get the edges in the BRep form

        Returns:
            List[TopoDS_Edge]: List of edges
        """
        edges = []
        for i, edge_sample_points in enumerate(self.EdgeSamplePoints):
            # 34 points
            gp_array = TColgp_Array1OfPnt(1, len(edge_sample_points))
            for index_i, sample_point in enumerate(edge_sample_points):
                gp_array.SetValue(index_i + 1, gp_Pnt(*sample_point))

            for precision in self._EDGE_FITTING_TOLERANCE:
                try:
                    bspline_edge = GeomAPI_PointsToBSpline(
                            gp_array,
                            self._CURVELENGTH,
                            self._CURVATURE,
                            self._TORSION,
                            # self._BSPLINE_CURVE_DEGREE_MIN,
                            self._BSPLINE_CURVE_DEGREE_MAX,
                            self._CONTINUITY,
                            precision
                    ).Curve()
                    edges.append(BRepBuilderAPI_MakeEdge(bspline_edge).Edge())
                    break
                except Exception as e:
                    print(e)
                    trimesh.PointCloud(edge_sample_points).export(f'{i}_edge.ply')
                    continue
        return edges

    def get_faces(self, tolerance) -> List[TopoDS_Face]:
        faces = []
        for i, face_sample_points in enumerate(self.FaceSamplePoints):
            u_dim = face_sample_points.shape[0]
            v_dim = face_sample_points.shape[1]
            gp_array = TColgp_Array2OfPnt(1, u_dim, 1, v_dim)
            for index_i in range(1, u_dim + 1):
                for index_j in range(1, v_dim + 1):
                    gp_array.SetValue(index_i, index_j, gp_Pnt(*face_sample_points[index_i - 1, index_j - 1]))

            for precision in self._FACE_FITTING_TOLERANCE:
                try:
                    bspline_face = GeomAPI_PointsToBSplineSurface(
                            gp_array,
                            self._CURVELENGTH,
                            self._CURVATURE,
                            self._TORSION,
                            # self._BSPLINE_CURVE_DEGREE_MIN,
                            self._BSPLINE_CURVE_DEGREE_MAX,
                            self._CONTINUITY,
                            precision
                    ).Surface()
                    faces.append(BRepBuilderAPI_MakeFace(bspline_face, tolerance).Face())
                    break
                except Exception as e:
                    print("Some face is not able to be fitted")
                    continue

        return faces

    @property
    def Vertices(self) -> List[TopoDS_Vertex]:
        """
        The list of TopoDS_Vertex
        """
        vertices = self._vertices
        return vertices

    @property
    def Edges(self) -> List[TopoDS_Edge]:
        """
        The list of TopoDS_Edge
        """
        edges = self._edges
        return edges

    @property
    def Faces(self) -> List[TopoDS_Face]:
        """
        The list of TopoDS_Face
        """
        faces = self._faces
        return faces

    @property
    def VertexPositions(self) -> np.ndarray:
        """
        Vertex positions

        Returns:
            np.ndarray: (num_vertices, 3)
        """
        vertex_sample_points = self._vertex_sample_points
        return vertex_sample_points

    @property
    def EdgeSamplePoints(self) -> np.ndarray:
        """
        Sample points on every edge

        Returns:
            np.ndarray: (num_edges, edge_sample_frequency, 3)
        """
        edge_sample_points = self._edge_sample_points
        return edge_sample_points

    @property
    def FaceSamplePoints(self) -> np.ndarray:
        """
        Sample points on every face

        Returns:
            np.ndarray: (num_faces, u_sample_frequency, v_sample_frequency, 3)
        """
        face_sample_points = self._face_sample_points
        return face_sample_points

    @property
    def FaceEdge(self) -> dict:
        face_edge = self._face_edge
        return face_edge

    @property
    def EdgeVertex(self) -> dict:
        edge_vertex = self._edge_vertex
        return edge_vertex

    @property
    def TransferTolerance(self):
        """
        The tolerance to build a TopoDS_Face
        """
        return self._transfer_tolerance

    @TransferTolerance.setter
    def TransferTolerance(self, value):
        self._transfer_tolerance = value
        self._rebuild_faces(self._transfer_tolerance)

    @property
    @abstractmethod
    def FaceSampleFrequency(self):
        pass

    @abstractmethod
    def _read_file(self, file_path: str | Path):
        pass

    @abstractmethod
    def _get_vertex_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_edge_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_face_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_face_edge(self) -> dict:
        pass

    @abstractmethod
    def _get_edge_vertex(self) -> dict:
        pass

    @abstractmethod
    def _openness_of_face(self, index: int, direction: int) -> bool:
        pass

    @abstractmethod
    def _openness_of_curve(self, index: int) -> bool:
        pass

    def _rebuild_faces(self):
        """
        Rebuild the sufaces when the tolerance is changed
        """
        self._faces = self.get_faces(self.TransferTolerance)


class ComplexGenProcessor(DataProcessor):

    def __init__(self, file_path: str | Path):
        super().__init__(file_path)

    def get_data(self, v_num_per_m: int):
        vertices, vertex_points = self.Vertices, self.VertexPositions
        edges, edge_points = self.sample_points_on_edges(v_num_per_m)
        faces, face_points = self.sample_points_on_faces(v_num_per_m)
        return faces, face_points, edges, edge_points, vertices, vertex_points

    def sample_points_on_faces(self, v_num_per_m: int):
        faces, face_points = [], []

        for face in self.Faces:
            try:
                v, f = get_triangulations(face, 0.1, 0.1)
                if len(f) == 0:
                    print("Ignore 0 face")
                    continue
            except:
                print("Ignore 1 face")
                continue
            mesh_item = trimesh.Trimesh(vertices=v, faces=f)
            area = mesh_item.area
            num_samples = min(max(int(v_num_per_m * v_num_per_m * area), 5), 10000)
            pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
            normals = mesh_item.face_normals[id_face]
            faces.append(face)
            face_points.append(np.concatenate((pc_item, normals), axis=1))

        return faces, face_points

    def sample_points_on_edges(self, v_num_per_m: int):
        edges, edge_points = [], []

        for edge in self.Edges:
            length = get_curve_length(edge)
            num_samples = min(max(int(v_num_per_m * length), 5), 10000)
            v = get_points_along_edge(edge, num_samples)
            edges.append(edge)
            edge_points.append(v)

        return edges, edge_points

    @property
    def FaceSampleFrequency(self):
        return int(np.sqrt(len(self.file_content['patches'][0]['grid']) / 3))

    def _read_file(self, file_path: str | Path) -> dict:
        """Return the raw file content from a JSON file

        Args:
            file_path (str | Path): file path

        Returns:
            dict: JSON dictionary
        """
        with open(file_path, 'r') as json_file:
            return json.load(json_file)

    def _get_vertex_array(self):
        vertices = []
        if self.file_content['corners'] is None:
            return np.asarray((0,0,0), dtype=np.float64)[None, None]
        for vertex_data in self.file_content['corners']:
            vertices.append(vertex_data['pts'])
        return np.array(vertices, dtype=np.float64)[:, None, :]

    def _get_edge_array(self):
        edges = []
        if self.file_content['curves'] is None:
            return np.array(edges, dtype=np.float64)
        for edge_data in self.file_content['curves']:
            edges.append(np.array(edge_data['pts']).reshape([-1, 3]))
        return np.array(edges, dtype=np.float64)

    def _get_face_array(self):
        faces = []
        if self.file_content['patches'] is None:
            return np.array(faces, dtype=np.float64)
        for face_data in self.file_content['patches']:
            faces.append(np.array(face_data['grid']).reshape([self.FaceSampleFrequency, self.FaceSampleFrequency, 3]))
        return np.array(faces, dtype=np.float64)

    def _openness_of_face(self, index: int, direction: int):
        if direction == self._U_AXIS:
            return self.file_content['patches'][index]['u_closed']
        elif direction == self._V_AXIS:
            return self.file_content['patches'][index]['v_closed']
        else:
            raise ValueError("Axis Error: Must be U_AXIS or V_AXIS!")

    def _openness_of_curve(self, index: int):
        return self.file_content['curves'][index]['closed']

    def _get_face_edge(self):
        face_edge_matrix = self.file_content['patch2curve']
        return self.__convert_topo_matrix(face_edge_matrix)

    def _get_edge_vertex(self):
        edge_vertex_matrix = self.file_content['curve2corner']
        return self.__convert_topo_matrix(edge_vertex_matrix)

    def __convert_topo_matrix(self, topo_matrix):
        topo_dict = dict()
        if topo_matrix is None:
            return topo_dict
        for main_index, topo_one_hot in enumerate(topo_matrix):
            connectivity = []
            for secondary_index, is_connected in enumerate(topo_one_hot):
                if is_connected == 1:
                    connectivity.append(secondary_index)
            if len(connectivity) > 0:
                topo_dict[main_index] = connectivity
        return topo_dict