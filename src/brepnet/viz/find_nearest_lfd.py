from lfd import LightFieldDistance
import trimesh

# rest of code
mesh_1: trimesh.Trimesh = trimesh.load("lfd/examples/cup1.obj")
mesh_2: trimesh.Trimesh = trimesh.load("lfd/examples/airplane.obj")

lfd_value: float = LightFieldDistance(verbose=True).get_distance(
        mesh_1.vertices, mesh_1.faces,
        mesh_2.vertices, mesh_2.faces
)
