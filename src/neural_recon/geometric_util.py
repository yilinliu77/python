import torch

def get_perpendicular_direction(v_dir):
    up_vector = np.array((0,0,1), dtype=np.float32)
    homo_dir = to_homogeneous_vector(v_dir)
    perpendicular_vector = np.cross(homo_dir, up_vector)[:,:2]
    return perpendicular_vector

def vectors_to_angles(normal_vectors):
    normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
    x, y, z = normal_vectors.unbind(dim=1)
    phi = torch.atan2(y, x)
    theta = torch.acos(z)
    return torch.stack([phi, theta], dim=1)


def angles_to_vectors(angles):
    phi, theta = angles.unbind(dim=-1)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


# planes: n*4 ray: n*3
def intersection_of_ray_and_plane(planes, rays_direction):
    n = planes[:, :3]
    d = planes[:, 3]
    denominator = torch.sum(n * rays_direction, dim=1)
    t = -d / torch.sum(n * rays_direction, dim=1)
    intersection_points = torch.unsqueeze(t, 1) * rays_direction
    valid_intersection = (denominator != 0) & (t >= 0)
    # n * _
    return valid_intersection, intersection_points


# planes: n*4 ray: m*3
def intersection_of_ray_and_all_plane(planes_abcd, vertexes_ray):
    n_planes = planes_abcd.size(0)
    m_rays = vertexes_ray.size(0)

    # 将平面参数和射线向量扩展为广播兼容的形状
    planes_abcd_expanded = planes_abcd.unsqueeze(1).expand(n_planes, m_rays, 4)
    vertexes_ray_expanded = vertexes_ray.unsqueeze(0).expand(n_planes, m_rays, 3)

    # 计算射线与平面的交点
    numerator = -planes_abcd_expanded[..., -1].unsqueeze(-1)  # 分子：-(D)
    denominator = torch.sum(planes_abcd_expanded[..., :3] * vertexes_ray_expanded, dim=-1).unsqueeze(
        -1)  # 分母：(A * v_ray.x + B * v_ray.y + C * v_ray.z)
    t = numerator / denominator  # t = -(D) / (A * v_ray.x + B * v_ray.y + C * v_ray.z)
    intersection_points = t * vertexes_ray_expanded  # p = p0 + t * v_ray，这里我们假设 p0 为原点 (0, 0, 0)
    # n*m*3
    return intersection_points


def compute_plane_abcd(patch_ray, ray_depth, normal):
    # patch_ray：n*3
    # ray_depth：n*100
    # normal:n*100*3
    intersection = patch_ray.unsqueeze(1).tile(1, 100, 1) * ray_depth[:, :, None]
    d = -torch.sum(intersection * normal, dim=-1, keepdim=True)
    plane_abcd = torch.cat([normal, d], dim=-1)
    return intersection, plane_abcd


def compute_area(v_points_2d):
    p1, p2, p3 = v_points_2d[:, 0], v_points_2d[:, 1], v_points_2d[:, 2]
    area = ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
            - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
            ).abs() / 2
    return area


# fun defined by xdt
def fit_plane_svd(points: torch.Tensor) -> torch.Tensor:
    centroid = torch.mean(points, axis=0)
    centered_points = points - centroid
    u, s, vh = torch.linalg.svd(centered_points)
    d = -torch.dot(vh[-1], centroid)
    abcd = torch.cat((vh[-1], torch.tensor([d]).to(points.device)))
    return abcd
