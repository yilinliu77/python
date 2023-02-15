import torch
import numpy as np

def sample_uniform(num_samples: int):
    """Sample uniformly in [-1,1] bounding volume.

    Args:
        num_samples(int) : number of points to sample
    """
    return torch.rand(num_samples, 3) * 2.0 - 1.0


def per_face_normals(
        V: torch.Tensor,
        F: torch.Tensor):
    """Compute normals per face.
    """
    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    normals = torch.cross(vec_a, vec_b)
    return normals


def area_weighted_distribution(
        V: torch.Tensor,
        F: torch.Tensor,
        normals: torch.Tensor = None):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    """

    if normals is None:
        normals = per_face_normals(V, F)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10

    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))


def sample_near_surface(
        V: torch.Tensor,
        F: torch.Tensor,
        num_samples: int,
        variance: float = 0.01,
        distrib=None):
    """Sample points near the mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)
    samples = sample_surface(V, F, num_samples, distrib)[0]
    samples += torch.randn_like(samples) * variance
    return samples


def random_face(
        V: torch.Tensor,
        F: torch.Tensor,
        num_samples: int,
        distrib=None):
    """Return an area weighted random sample of faces and their normals from the mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): num of samples to return
        distrib: distribution to use. By default, area-weighted distribution is used.
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    normals = per_face_normals(V, F)

    idx = distrib.sample([num_samples])

    return F[idx], normals[idx]


def sample_surface(
        V: torch.Tensor,
        F: torch.Tensor,
        num_samples: int,
        distrib=None):
    """Sample points and their normals on mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    # Select faces & sample their surface
    fidx, normals = random_face(V, F, num_samples, distrib)
    f = V[fidx]

    u = torch.sqrt(torch.rand(num_samples)).to(V.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(V.device).unsqueeze(-1)

    samples = (1 - u) * f[:, 0, :] + (u * (1 - v)) * f[:, 1, :] + u * v * f[:, 2, :]

    return samples, normals

def sample_points_cpu(v_vertices, v_faces,
                      v_num_sample_uniform, v_num_sample_on_surface,v_numsample_near_surface):
    with torch.no_grad():
        vertices = torch.tensor(v_vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(v_faces, dtype=torch.long, device="cuda")

        samples = []
        distrib = area_weighted_distribution(vertices, faces)
        samples.append(sample_surface(vertices, faces, v_num_sample_on_surface, distrib=distrib)[0])
        samples.append(sample_near_surface(vertices, faces, v_numsample_near_surface, distrib=distrib))
        samples.append(sample_uniform(v_num_sample_uniform).to(vertices.device))
        samples_points = torch.cat(samples, dim=0).contiguous()
    return samples_points