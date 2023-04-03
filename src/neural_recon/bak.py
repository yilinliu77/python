class LModel0(nn.Module):
    def __init__(self, v_data):
        super(LModel0, self).__init__()
        self.seg_distance_normalizer = 300
        self.register_buffer("ray_c", torch.tensor(
            [v_data["graph1"].nodes[id_node]["ray_c"].tolist() for id_node in v_data["graph1"].nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor(
                                 [v_data["graph1"].nodes[id_node]["distance"] for id_node in v_data["graph1"].nodes()],
                                 dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        self.length_normalizer = 5
        self.vertical_length1 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length2 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length1 /= self.length_normalizer
        self.vertical_length2 /= self.length_normalizer

        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up_dict = {}
        for edge in v_data["graph1"].edges():
            pass
        self.v_up1 = torch.tensor(
            [v_data["graph1"].nodes[id_node]["ray_c"].tolist() for id_node in v_data["graph1"].nodes()],
            dtype=torch.float32)
        self.v_up2 = -torch.tensor(v_data["v_up_c"]).float()
        self.v_up1 = vector_to_sphere_coordinate(self.v_up1)
        self.v_up2 = vector_to_sphere_coordinate(self.v_up2)
        self.v_up1[0] = self.v_up1[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up2[0] = self.v_up2[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up1[1] = self.v_up1[1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up2[1] = self.v_up2[1] / self.theta_normalizer  # [0,pi] -> [0,1]

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=False)
        self.v_up1 = nn.Parameter(self.v_up1, requires_grad=True)
        self.v_up2 = nn.Parameter(self.v_up2, requires_grad=True)
        self.vertical_length1 = nn.Parameter(self.vertical_length1, requires_grad=False)
        self.vertical_length2 = nn.Parameter(self.vertical_length2, requires_grad=False)

        self.seg_2d = torch.tensor(v_data["seg2d"]).float().cuda()
        self.ray_c = torch.tensor(v_data["ray_c"]).float().cuda()
        self.intrinsic1 = torch.as_tensor(v_data["intrinsic1"]).float().cuda()
        self.intrinsic2 = torch.as_tensor(v_data["intrinsic2"]).float().cuda()
        self.extrinsic1 = torch.as_tensor(v_data["extrinsic1"]).float().cuda()
        self.extrinsic2 = torch.as_tensor(v_data["extrinsic2"]).float().cuda()

        self.img_model1 = v_data["img_model1"]
        self.img_model1.freeze()
        self.img_model2 = v_data["img_model2"]
        self.img_model2.freeze()

        # Visualization
        self.rgb1 = v_data["rgb1"]
        self.rgb2 = v_data["rgb2"]

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        v_up11 = (self.v_up1[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up1[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        vertical_length1 = self.vertical_length1 * self.length_normalizer
        vertical_length2 = self.vertical_length2 * self.length_normalizer
        return seg_distance, v_up, vertical_length1, vertical_length2

    def debug_save(self, v_index):
        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up[:, 0]

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
                                                               cylinder_height=0.00025, cone_height=0.00025,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])

    # Out of date
    def find_up_vector(self, id_patch, id_start, id_end, v_up):
        id_up = -1
        if (id_start, id_end) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_start, id_end)]
        elif (id_end, id_start) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_end, id_start)]
        else:
            raise
        return v_up[id_up]

    def compute_similarity_loss(self,
                                start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                v_is_debug, v_is_log, v_log_frequency, v_index, id_segment
                                ):
        roi_bound1, roi_coor1, roi_2d1 = compute_roi(
            start_c, end_c, (start_c + end_c) / 2,
            cur_dir, v_up_c, vertical_length1,
            self.intrinsic1
        )

        # Visualize region
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # Selected segment
            cv2.line(line_img1,
                     (self.graph1.nodes[id_start]["pos_2d"] * shape).astype(np.int32),
                     (self.graph1.nodes[id_end]["pos_2d"] * shape).astype(np.int32),
                     (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # RoI
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (255, 0, 0),
                                      thickness=1, lineType=cv2.LINE_AA)

            # cv2.circle(line_img1, (roi_2d_numpy[0] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[1] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[3] * shape).astype(np.int32), 10, (0, 0, 255), 10)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{}_{:05d}.jpg".format(id_segment, v_index),
                        line_img1)
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", line_img1)
                cv2.waitKey()

        roi_coor_shape = roi_coor1.shape
        roi_coor_2d = torch.transpose(self.intrinsic1 @ torch.transpose(roi_coor1.reshape((-1, 3)), 0, 1), 0, 1)
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        roi_coor_2d = roi_coor_2d.reshape(roi_coor_shape[:2] + (2,))

        sampled_img = sample_img_prediction(self.img_model1, roi_coor_2d)
        # sampled_img = sample_img(self.o_rgb1, roi_coor_2d)

        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        roi_coor_2d_img2 = torch.transpose(
            transformation @ torch.transpose(to_homogeneous_tensor(roi_coor1.reshape((-1, 3))), 0, 1), 0, 1)
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        roi_coor_2d_img2 = roi_coor_2d_img2.reshape(roi_coor_shape[:2] + (2,))
        viz_sampled_img2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2)
        # viz_sampled_img2 = sample_img(self.o_rgb2, roi_coor_2d_img2)

        # Visualize query points
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            # line_img1 = cv2.resize(line_img1, (600, 400))
            shape = line_img1.shape[:2][::-1]

            roi_coor_2d1_numpy = roi_coor_2d.detach().cpu().numpy()
            sampled_img1_numpy = normalized_torch_img_to_numpy(sampled_img.permute(2, 0, 1))
            roi_coor_2d2_numpy = roi_coor_2d_img2.detach().cpu().numpy()
            sampled_img2_numpy = normalized_torch_img_to_numpy(viz_sampled_img2.permute(2, 0, 1))

            for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img1 = pad_and_enlarge_along_y(sampled_img1_numpy, line_img1)
            line_img2 = self.rgb2.copy()
            # line_img2 = cv2.resize(line_img2, (600, 400))
            shape = line_img2.shape[:2][::-1]
            for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img2 = pad_and_enlarge_along_y(sampled_img2_numpy, line_img2)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(id_segment, v_index),
                        np.concatenate((img1, img2), axis=0))
            if v_is_debug:
                print("Visualize the extracted region")
                cv2.imshow("1", np.concatenate((img1, img2), axis=0))
                cv2.waitKey()

        loss = torch.nn.functional.mse_loss(sampled_img, viz_sampled_img2)
        return loss

    def compute_normal_consistency(self,
                                   point_pos_c, id1, id2, v_ref_normal, v_target_normal
                                   ):
        point1 = point_pos_c[id1]
        point2 = point_pos_c[id2]
        cur_dir = normalize_tensor(point2 - point1)

        v_ref_normal = normalize_tensor(v_ref_normal)
        v_next_normal = normalize_tensor(v_target_normal)

        loss = (1 - v_next_normal.dot(v_ref_normal)) / 2  # [0, 2] -> [0, 1]
        return loss

    def forward(self, v_index):
        v_is_debug = False
        v_is_log = True
        v_log_frequency = 10000

        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()

        point_pos_c = self.ray_c * seg_distance[:, None]
        normal_losses = []
        similarity_losses = []
        time_similarity = 0
        time_consistency = 0
        for face_ids in self.graph1.graph["faces"][self.id_patch, self.id_patch + 1]:
            # for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]

                start_c = point_pos_c[id_start]
                end_c = point_pos_c[id_end]

                cur_dir = normalize_tensor(end_c - start_c)

                v_up_c = self.find_up_vector(id_patch, id_start, id_end, v_up)

                a = time.time()
                loss = self.compute_similarity_loss(start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                                    v_is_debug and id_patch == self.id_viz_patch,
                                                    v_is_log and id_patch == self.id_viz_patch, v_log_frequency,
                                                    v_index,
                                                    id_segment)
                time_similarity += time.time() - a
                a = time.time()
                # Normal loss
                v_cur_normal = torch.cross(cur_dir, v_up_c)
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]

                v_next_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_end, id_next, v_up))
                v_prev_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_prev, id_start, v_up))
                normal_loss1 = self.compute_normal_consistency(point_pos_c, id_prev, id_start, v_cur_normal,
                                                               v_prev_normal)
                normal_loss2 = self.compute_normal_consistency(point_pos_c, id_end, id_next, v_cur_normal,
                                                               v_next_normal)
                normal_loss = (normal_loss1 + normal_loss2) / 2
                time_consistency += time.time() - a
                normal_losses.append(normal_loss)
                similarity_losses.append(loss)

        normal_losses = torch.stack(normal_losses).mean()
        similarity_losses = torch.stack(similarity_losses).mean()
        total_loss = normal_losses * 0.5 + similarity_losses * 0.5
        return total_loss, normal_losses, similarity_losses


class LModel2(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel2, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        v_up_c = []
        # self.v_up_dict = {item: {} for item in range(len(self.graph1.graph["faces"]))}
        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                # self.v_up_dict[id_patch][id_segment] = len(v_up_c)
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                up_c = self.graph1.edges[(face_ids[id_segment], id_end)]["up_c"][id_patch]
                v_up_c.append(up_c.tolist())
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)

        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up = torch.tensor(v_up_c).float()
        self.v_up = vector_to_sphere_coordinate(self.v_up)
        self.v_up[:, 0] = self.v_up[:, 0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up[:, 1] = self.v_up[:, 1] / self.theta_normalizer  # [0,pi] -> [0,1]

        self.length_normalizer = 5
        self.vertical_length1 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length2 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length1 /= self.length_normalizer
        self.vertical_length2 /= self.length_normalizer

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)
        self.vertical_length1 = nn.Parameter(self.vertical_length1, requires_grad=False)
        self.vertical_length2 = nn.Parameter(self.vertical_length2, requires_grad=False)
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)

        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def __init__1(self, v_data):
        super(LModel, self).__init__()
        self.seg_distance_normalizer = 300
        self.seg_distance = torch.tensor(v_data["seg_distances"]).float()
        self.seg_distance /= self.seg_distance_normalizer

        self.length_normalizer = 5
        self.vertical_length1 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length2 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length1 /= self.length_normalizer
        self.vertical_length2 /= self.length_normalizer

        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up1 = torch.tensor(v_data["v_up_c"]).float()
        self.v_up2 = -torch.tensor(v_data["v_up_c"]).float()
        self.v_up1 = vector_to_sphere_coordinate(self.v_up1)
        self.v_up2 = vector_to_sphere_coordinate(self.v_up2)
        self.v_up1[0] = self.v_up1[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up2[0] = self.v_up2[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up1[1] = self.v_up1[1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up2[1] = self.v_up2[1] / self.theta_normalizer  # [0,pi] -> [0,1]

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=False)
        self.v_up1 = nn.Parameter(self.v_up1, requires_grad=True)
        self.v_up2 = nn.Parameter(self.v_up2, requires_grad=True)
        self.vertical_length1 = nn.Parameter(self.vertical_length1, requires_grad=False)
        self.vertical_length2 = nn.Parameter(self.vertical_length2, requires_grad=False)

        self.seg_2d = torch.tensor(v_data["seg2d"]).float().cuda()
        self.ray_c = torch.tensor(v_data["ray_c"]).float().cuda()
        self.intrinsic1 = torch.as_tensor(v_data["intrinsic1"]).float().cuda()
        self.intrinsic2 = torch.as_tensor(v_data["intrinsic2"]).float().cuda()
        self.extrinsic1 = torch.as_tensor(v_data["extrinsic1"]).float().cuda()
        self.extrinsic2 = torch.as_tensor(v_data["extrinsic2"]).float().cuda()

        self.img_model1 = v_data["img_model1"]
        self.img_model1.freeze()
        self.img_model2 = v_data["img_model2"]
        self.img_model2.freeze()

        # Visualization
        self.rgb1 = v_data["rgb1"]
        self.rgb2 = v_data["rgb2"]

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        v_up11 = (self.v_up[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        vertical_length1 = self.vertical_length1 * self.length_normalizer
        vertical_length2 = self.vertical_length2 * self.length_normalizer
        return seg_distance, v_up, vertical_length1, vertical_length2

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()

        # 0: Unpack data
        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        edge_points = point_pos_c[point_index].reshape((-1, 4, 3))
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        next_point = edge_points[:, 2]
        prev_point = edge_points[:, 3]
        # up_cur = v_up[point_index].reshape((-1, 4, 3))[:, 0]
        # up_next = v_up[point_index].reshape((-1, 4, 3))[:, 3]
        # up_prev = v_up[point_index].reshape((-1, 4, 3))[:, 2]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        up_vectors = []
        next_dir = next_point - end_point
        normal_cur = torch.cross(cur_dir, next_dir)
        up_cur1 = normalize_tensor(torch.cross(normal_cur, cur_dir))
        up_vectors.append(up_cur1)
        prev_dir = start_point - prev_point
        normal_prev = torch.cross(prev_dir, cur_dir)
        up_cur2 = normalize_tensor(torch.cross(normal_prev, cur_dir))
        up_vectors.append(up_cur2)

        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = vertical_length1  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - begin_idxes.repeat_interleave(
            num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(
            num_horizontal) * half_window_size_meter_horizontal.repeat_interleave(
            num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        similarity_losses = []
        for i in range(2):
            # Local -> Global image coordinates
            interpolated_coordinates_camera = cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:,
                                                                                                           None] \
                                              + up_vectors[i].repeat_interleave(num_coordinates_per_edge,
                                                                                dim=0) * coords_y[:,
                                                                                         None] \
                                              + start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
            time_profile[3], timer = refresh_timer(timer)

            roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
            roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
            valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
            valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
            roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
            time_profile[4], timer = refresh_timer(timer)
            if self.img_method == "model":
                sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
            else:
                sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
            time_profile[5], timer = refresh_timer(timer)

            # Second img
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
            roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
            valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
            valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
            roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
            if self.img_method == "model":
                sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
            else:
                sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
            time_profile[6], timer = refresh_timer(timer)

            similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
            time_profile[7], timer = refresh_timer(timer)
            similarity_losses.append(similarity_loss)

        # 8: Normal loss
        normal_loss = (1 - (up_vectors[0] * up_vectors[1]).sum(dim=1).mean()) / 2  # [0, 2] -> [0, 1]
        time_profile[8], timer = refresh_timer(timer)

        # total_loss = normal_loss * 0.5 + similarity_loss * 0.5
        similarity_loss = torch.mean(torch.stack(similarity_losses))
        total_loss = normal_loss * self.loss_weights[0] + similarity_loss * self.loss_weights[1]

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_start = self.graph1.graph["faces"][self.id_viz_patch][idx]
                id_end = self.graph1.graph["faces"][self.id_viz_patch][
                    (idx + 1) % len(self.graph1.graph["faces"][self.id_viz_patch])]
                line_img1 = line_img1_base.copy()
                # Selected segment
                cv2.line(line_img1,
                         (self.graph1.nodes[id_start]["pos_2d"] * shape).astype(np.int32),
                         (self.graph1.nodes[id_end]["pos_2d"] * shape).astype(np.int32),
                         (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

                id_magic = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                # RoI
                # Compute extreme point
                point1 = start_point[id_magic]
                point2 = start_point[id_magic] + up_vectors[0][id_magic] * half_window_size_meter_vertical
                point3 = end_point[id_magic] + up_vectors[0][id_magic] * half_window_size_meter_vertical
                point4 = end_point[id_magic]
                roi = torch.stack((point1, point2, point3, point4), dim=0)
                roi_2d1 = torch.transpose(self.intrinsic1 @ torch.transpose(roi, 0, 1), 0, 1)
                roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]

                roi_2d_numpy = roi_2d1.detach().cpu().numpy()
                line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (0, 0, 255),
                                          thickness=2, lineType=cv2.LINE_AA)

                roi_2d2 = (transformation @ to_homogeneous_tensor(roi).T).T
                roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
                line_img2 = self.rgb2.copy()
                roi_2d_numpy = roi_2d2.detach().cpu().numpy()
                line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (0, 0, 255),
                                          thickness=2, lineType=cv2.LINE_AA)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the calculated roi")
                    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("1", 1600, 900)
                    cv2.moveWindow("1", 5, 5)
                    cv2.imshow("1", line_img1)
                    cv2.waitKey()

                # Visualize query points
                # For pixel coordinate
                id_pixel_start = torch.sum(num_coordinates_per_edge[:id_magic])
                w = num_horizontal[id_magic]
                h = num_vertical

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                roi_coor_2d1_numpy = roi_coor_2d[id_pixel_start:id_pixel_start + h * w].detach().cpu().numpy()
                sampled_img1_numpy = normalized_torch_img_to_numpy(
                    sample_imgs1[id_pixel_start:id_pixel_start + h * w].reshape((w, h, 3)).transpose(0, 1).permute(2, 0,
                                                                                                                   1))
                roi_coor_2d2_numpy = roi_coor_2d_img2[id_pixel_start:id_pixel_start + h * w].detach().cpu().numpy()
                sampled_img2_numpy = normalized_torch_img_to_numpy(
                    sample_imgs2[id_pixel_start:id_pixel_start + h * w].reshape((w, h, 3)).transpose(0, 1).permute(2, 0,
                                                                                                                   1))

                for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                img1 = pad_and_enlarge_along_y(sampled_img1_numpy, line_img1)
                line_img2 = self.rgb2.copy()
                # line_img2 = cv2.resize(line_img2, (600, 400))
                shape = line_img2.shape[:2][::-1]
                for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                    cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                img2 = pad_and_enlarge_along_y(sampled_img2_numpy, line_img2)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((img1, img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((sampled_img1_numpy, sampled_img2_numpy), axis=0))
                    cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up[:, 0]

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
                                                               cylinder_height=0.00025, cone_height=0.00025,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])

    # Out of date
    def denormalize2(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer

        vertical_length1 = self.vertical_length1 * self.length_normalizer
        vertical_length2 = self.vertical_length2 * self.length_normalizer
        return seg_distance, vertical_length1, vertical_length2

    def find_up_vector(self, id_patch, id_start, id_end, v_up):
        id_up = -1
        if (id_start, id_end) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_start, id_end)]
        elif (id_end, id_start) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_end, id_start)]
        else:
            raise
        return v_up[id_up]

    def compute_similarity_loss(self,
                                start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                v_is_debug, v_is_log, v_log_frequency, v_index, id_segment
                                ):
        roi_bound1, roi_coor1, roi_2d1 = compute_roi(
            start_c, end_c, (start_c + end_c) / 2,
            cur_dir, v_up_c, vertical_length1,
            self.intrinsic1
        )

        # Visualize region
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # Selected segment
            cv2.line(line_img1,
                     (self.graph1.nodes[id_start]["pos_2d"] * shape).astype(np.int32),
                     (self.graph1.nodes[id_end]["pos_2d"] * shape).astype(np.int32),
                     (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # RoI
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (255, 0, 0),
                                      thickness=1, lineType=cv2.LINE_AA)

            # cv2.circle(line_img1, (roi_2d_numpy[0] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[1] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[3] * shape).astype(np.int32), 10, (0, 0, 255), 10)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{}_{:05d}.jpg".format(id_segment, v_index),
                        line_img1)
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", line_img1)
                cv2.waitKey()

        roi_coor_shape = roi_coor1.shape
        roi_coor_2d = torch.transpose(self.intrinsic1 @ torch.transpose(roi_coor1.reshape((-1, 3)), 0, 1), 0, 1)
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        roi_coor_2d = roi_coor_2d.reshape(roi_coor_shape[:2] + (2,))

        sampled_img = sample_img_prediction(self.img_model1, roi_coor_2d)
        # sampled_img = sample_img(self.o_rgb1, roi_coor_2d)

        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        roi_coor_2d_img2 = torch.transpose(
            transformation @ torch.transpose(to_homogeneous_tensor(roi_coor1.reshape((-1, 3))), 0, 1), 0, 1)
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        roi_coor_2d_img2 = roi_coor_2d_img2.reshape(roi_coor_shape[:2] + (2,))
        viz_sampled_img2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2)
        # viz_sampled_img2 = sample_img(self.o_rgb2, roi_coor_2d_img2)

        # Visualize query points
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            # line_img1 = cv2.resize(line_img1, (600, 400))
            shape = line_img1.shape[:2][::-1]

            roi_coor_2d1_numpy = roi_coor_2d.detach().cpu().numpy()
            sampled_img1_numpy = normalized_torch_img_to_numpy(sampled_img.permute(2, 0, 1))
            roi_coor_2d2_numpy = roi_coor_2d_img2.detach().cpu().numpy()
            sampled_img2_numpy = normalized_torch_img_to_numpy(viz_sampled_img2.permute(2, 0, 1))

            for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img1 = pad_and_enlarge_along_y(sampled_img1_numpy, line_img1)
            line_img2 = self.rgb2.copy()
            # line_img2 = cv2.resize(line_img2, (600, 400))
            shape = line_img2.shape[:2][::-1]
            for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img2 = pad_and_enlarge_along_y(sampled_img2_numpy, line_img2)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(id_segment, v_index),
                        np.concatenate((img1, img2), axis=0))
            if v_is_debug:
                print("Visualize the extracted region")
                cv2.imshow("1", np.concatenate((img1, img2), axis=0))
                cv2.waitKey()

        loss = torch.nn.functional.mse_loss(sampled_img, viz_sampled_img2)
        return loss

    def compute_normal_consistency(self,
                                   point_pos_c, id1, id2, v_ref_normal, v_target_normal
                                   ):
        point1 = point_pos_c[id1]
        point2 = point_pos_c[id2]
        cur_dir = normalize_tensor(point2 - point1)

        v_ref_normal = normalize_tensor(v_ref_normal)
        v_next_normal = normalize_tensor(v_target_normal)

        loss = (1 - v_next_normal.dot(v_ref_normal)) / 2  # [0, 2] -> [0, 1]
        return loss

    def forward_(self, v_index):
        v_is_debug = False
        v_is_log = True
        v_log_frequency = 10000

        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()

        point_pos_c = self.ray_c * seg_distance[:, None]
        normal_losses = []
        similarity_losses = []
        time_similarity = 0
        time_consistency = 0
        # for face_ids in self.graph1.graph["faces"][self.id_patch,self.id_patch+1]:
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]

                start_c = point_pos_c[id_start]
                end_c = point_pos_c[id_end]

                cur_dir = normalize_tensor(end_c - start_c)

                v_up_c = self.find_up_vector(id_patch, id_start, id_end, v_up)

                a = time.time()
                loss = self.compute_similarity_loss(start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                                    v_is_debug and id_patch == self.id_viz_patch,
                                                    v_is_log and id_patch == self.id_viz_patch, v_log_frequency,
                                                    v_index,
                                                    id_segment)
                time_similarity += time.time() - a
                a = time.time()
                # Normal loss
                v_cur_normal = torch.cross(cur_dir, v_up_c)
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]

                v_next_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_end, id_next, v_up))
                v_prev_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_prev, id_start, v_up))
                normal_loss1 = self.compute_normal_consistency(point_pos_c, id_prev, id_start, v_cur_normal,
                                                               v_prev_normal)
                normal_loss2 = self.compute_normal_consistency(point_pos_c, id_end, id_next, v_cur_normal,
                                                               v_next_normal)
                normal_loss = (normal_loss1 + normal_loss2) / 2
                time_consistency += time.time() - a
                normal_losses.append(normal_loss)
                similarity_losses.append(loss)

        normal_losses = torch.stack(normal_losses).mean()
        similarity_losses = torch.stack(similarity_losses).mean()
        total_loss = normal_losses * 0.5 + similarity_losses * 0.5
        return total_loss, normal_losses, similarity_losses


# Normal loss and Similarity loss using sample points inside polygon
class LModel3(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel3, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        return seg_distance

    def sample_points(self, v_edge_points):
        device = v_edge_points.device
        start_point = v_edge_points[:, 0]
        end_point = v_edge_points[:, 1]
        prev_point = v_edge_points[:, 2]
        next_point = v_edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point

        # Sample points on edges
        length = torch.linalg.norm(cur_dir, dim=1)
        num_per_edge_m2 = 100
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 10000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)

        # Sample points within triangle
        num_per_half_m2 = 2500
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir), dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 10000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1])
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        return [num_edge_points, num_polygon_points], [sampled_edge_points, sampled_polygon_points, ]

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        seg_distance = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        edge_points = point_pos_c[point_index].reshape((-1, 4, 3))
        time_profile[0], timer = refresh_timer(timer)

        # Sample points on edges
        [num_edge_points, num_polygon_points], [sampled_edge_points, sampled_polygon_points] = self.sample_points(
            edge_points)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        # Normal loss
        # index = torch.randint(0, coordinates.shape[0] - 1, (100,))
        # A = torch.cat((coordinates[index, :2], torch.ones_like((coordinates[index,0:1]))), dim=1) / self.seg_distance_normalizer
        # B = coordinates[index,2:] / self.seg_distance_normalizer
        # plane_equation = torch.linalg.lstsq(A,B).solution
        # normal_loss = (B - A @ plane_equation[:3]).pow(2).sum()
        normal_loss = similarity_loss
        total_loss = similarity_loss

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            # Visualize query points
            # For edges

            pc = o3d.geometry.PointCloud()
            viz_edge_points = coordinates
            pc.points = o3d.utility.Vector3dVector(
                (torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(coordinates).T).T[:, :3].cpu().numpy()
            )
            o3d.io.write_point_cloud("output/img_field_test/imgs_log/pc_{:05d}.ply".format(v_id_epoch), pc)

            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]

            roi_coor_2d1_numpy = (roi_coor_2d.detach().cpu().numpy() * shape).astype(np.int32)
            roi_coor_2d2_numpy = (roi_coor_2d_img2.detach().cpu().numpy() * shape).astype(np.int32)
            line_img1[roi_coor_2d1_numpy[:, 1], roi_coor_2d1_numpy[:, 0]] = (0, 0, 255)
            line_img2[roi_coor_2d2_numpy[:, 1], roi_coor_2d2_numpy[:, 0]] = (0, 0, 255)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the extracted region")
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            # for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
            #     id_edge = idx + len(
            #         list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4
            #
            #     # Visualize query points
            #     # For edges
            #     id_edge_points_start = torch.sum(num_edge_points[:id_edge])
            #     id_polygon_points_start = torch.sum(num_polygon_points[:id_edge]) + torch.sum(num_edge_points)
            #
            #     pc = o3d.geometry.PointCloud()
            #     viz_edge_points = coordinates[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]]
            #     viz_polygon_points = coordinates[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     pc.points = o3d.utility.Vector3dVector(
            #         (torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(
            #             torch.cat((viz_edge_points, viz_polygon_points), dim=0)).T).T[:, :3].cpu().numpy()
            #     )
            #     o3d.io.write_point_cloud("output/img_field_test/imgs_log/pc_{}_{:05d}.ply".format(idx, v_id_epoch), pc)
            #
            #     line_img1 = self.rgb1.copy()
            #     shape = line_img1.shape[:2][::-1]
            #
            #     roi_coor_2d1_numpy = torch.cat([
            #         roi_coor_2d[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]],
            #         roi_coor_2d[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     ]).detach().cpu().numpy()
            #     roi_coor_2d2_numpy = torch.cat([
            #         roi_coor_2d_img2[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]],
            #         roi_coor_2d_img2[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     ]).detach().cpu().numpy()
            #
            #     for item in roi_coor_2d1_numpy.reshape((-1, 2)):
            #         cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            #     line_img2 = self.rgb2.copy()
            #     shape = line_img2.shape[:2][::-1]
            #     for item in roi_coor_2d2_numpy.reshape((-1, 2)):
            #         cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            #
            #     cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
            #                 np.concatenate((line_img1, line_img2), axis=0))
            #     if v_is_debug:
            #         print("Visualize the extracted region")
            #         cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
            #         cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            # total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            # center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            # up_point = center_point_c + total_up[:, 0]
            #
            # center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
            #                  :3].cpu().numpy()
            # up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
            #                                :3].cpu().numpy() - center_point_w)
            #
            arrows = o3d.geometry.TriangleMesh()
            # for i in range(center_point_w.shape[0]):
            #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
            #                                                    cylinder_height=0.00025, cone_height=0.00025,
            #                                                    resolution=3, cylinder_split=1)
            #     arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
            #     arrow.translate(center_point_w[i])
            #     arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        # o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        # lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        # o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        # o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using sample points inside polygon and gaussian distribution
class LModel31(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel31, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)

        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def denormalize(self):
        # seg_distance = self.seg_distance * self.seg_distance_normalizer
        eps = torch.tensor(np.random.normal(0, 1, (10, self.gaussian_mean.shape[0])), dtype=torch.float32).to(
            self.gaussian_mean.device)
        eps = torch.clamp_min(eps, 1e-4)
        seg_distance = (self.gaussian_mean[None, :] + eps * self.gaussian_std[None, :]) * self.seg_distance_normalizer
        return seg_distance

    def sample_points(self, v_edge_points):
        device = v_edge_points.device
        start_point = v_edge_points[:, 0]
        end_point = v_edge_points[:, 1]
        prev_point = v_edge_points[:, 2]
        next_point = v_edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point

        # Sample points on edges
        length = torch.linalg.norm(cur_dir, dim=1)
        num_per_edge_m2 = 100
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 10000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)

        # Sample points within triangle
        num_per_half_m2 = 2500
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir), dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 10000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1])
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        return [num_edge_points, num_polygon_points], [sampled_edge_points, sampled_polygon_points, ]

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = 10
        num_vertices = [len(self.edge_point_index[item]) // 4 for item in v_index]
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        num_vertices = len(point_index) // 4
        seg_distance = self.denormalize()
        target_seg_distance = seg_distance[:, point_index].reshape(-1)
        point_index = point_index * seg_distance.shape[0]
        ray_c = self.ray_c[point_index]
        point_pos_c = ray_c * target_seg_distance[:, None]
        edge_points = point_pos_c.reshape((-1, 4, 3))
        time_profile[0], timer = refresh_timer(timer)

        # Sample points on edges
        [num_edge_points, num_polygon_points], [sampled_edge_points, sampled_polygon_points] = self.sample_points(
            edge_points)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        # Normal loss
        # index = torch.randint(0, coordinates.shape[0] - 1, (100,))
        # A = torch.cat((coordinates[index, :2], torch.ones_like((coordinates[index,0:1]))), dim=1) / self.seg_distance_normalizer
        # B = coordinates[index,2:] / self.seg_distance_normalizer
        # plane_equation = torch.linalg.lstsq(A,B).solution
        # normal_loss = (B - A @ plane_equation[:3]).pow(2).sum()
        normal_loss = similarity_loss
        total_loss = similarity_loss

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            # Visualize query points
            # For edges

            pc = o3d.geometry.PointCloud()
            viz_edge_points = coordinates
            pc.points = o3d.utility.Vector3dVector(
                (torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(coordinates).T).T[:, :3].cpu().numpy()
            )
            o3d.io.write_point_cloud("output/img_field_test/imgs_log/pc_{:05d}.ply".format(v_id_epoch), pc)

            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]

            roi_coor_2d1_numpy = (roi_coor_2d.detach().cpu().numpy() * shape).astype(np.int32)
            roi_coor_2d2_numpy = (roi_coor_2d_img2.detach().cpu().numpy() * shape).astype(np.int32)
            line_img1[roi_coor_2d1_numpy[:, 1], roi_coor_2d1_numpy[:, 0]] = (0, 0, 255)
            line_img2[roi_coor_2d2_numpy[:, 1], roi_coor_2d2_numpy[:, 0]] = (0, 0, 255)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the extracted region")
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            # for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
            #     id_edge = idx + len(
            #         list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4
            #
            #     # Visualize query points
            #     # For edges
            #     id_edge_points_start = torch.sum(num_edge_points[:id_edge])
            #     id_polygon_points_start = torch.sum(num_polygon_points[:id_edge]) + torch.sum(num_edge_points)
            #
            #     pc = o3d.geometry.PointCloud()
            #     viz_edge_points = coordinates[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]]
            #     viz_polygon_points = coordinates[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     pc.points = o3d.utility.Vector3dVector(
            #         (torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(
            #             torch.cat((viz_edge_points, viz_polygon_points), dim=0)).T).T[:, :3].cpu().numpy()
            #     )
            #     o3d.io.write_point_cloud("output/img_field_test/imgs_log/pc_{}_{:05d}.ply".format(idx, v_id_epoch), pc)
            #
            #     line_img1 = self.rgb1.copy()
            #     shape = line_img1.shape[:2][::-1]
            #
            #     roi_coor_2d1_numpy = torch.cat([
            #         roi_coor_2d[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]],
            #         roi_coor_2d[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     ]).detach().cpu().numpy()
            #     roi_coor_2d2_numpy = torch.cat([
            #         roi_coor_2d_img2[id_edge_points_start:id_edge_points_start + num_edge_points[id_edge]],
            #         roi_coor_2d_img2[id_polygon_points_start:id_polygon_points_start + num_polygon_points[id_edge]]
            #     ]).detach().cpu().numpy()
            #
            #     for item in roi_coor_2d1_numpy.reshape((-1, 2)):
            #         cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            #     line_img2 = self.rgb2.copy()
            #     shape = line_img2.shape[:2][::-1]
            #     for item in roi_coor_2d2_numpy.reshape((-1, 2)):
            #         cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            #
            #     cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
            #                 np.concatenate((line_img1, line_img2), axis=0))
            #     if v_is_debug:
            #         print("Visualize the extracted region")
            #         cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
            #         cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            # total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            # center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            # up_point = center_point_c + total_up[:, 0]
            #
            # center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
            #                  :3].cpu().numpy()
            # up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
            #                                :3].cpu().numpy() - center_point_w)
            #
            arrows = o3d.geometry.TriangleMesh()
            # for i in range(center_point_w.shape[0]):
            #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
            #                                                    cylinder_height=0.00025, cone_height=0.00025,
            #                                                    resolution=3, cylinder_split=1)
            #     arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
            #     arrow.translate(center_point_w[i])
            #     arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        # o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        # lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        # o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        # o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using up vector
class LModel11(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel11, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        v_up = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up = vector_to_sphere_coordinate(self.v_up)
        self.v_up[:, 0] = self.v_up[:, 0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up[:, 1] = self.v_up[:, 1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)
        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        v_up11 = (self.v_up[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        return seg_distance, v_up

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        seg_distance, v_up = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        edge_up_c = v_up[point_index].reshape((-1, 4, 3))
        edge_points = point_pos_c[point_index].reshape((-1, 4, 3))
        time_profile[0], timer = refresh_timer(timer)

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        similarity_losses = []
        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            edge_up_c[:, 0, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)
        similarity_losses.append(similarity_loss)

        # 8: Normal loss

        cur_normal = normalize_tensor(torch.cross(cur_dir, edge_up_c[:, 0]))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_normal = normalize_tensor(torch.cross(next_dir, edge_up_c[:, 1]))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_normal = normalize_tensor(torch.cross(prev_dir, edge_up_c[:, 2]))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2
        time_profile[8], timer = refresh_timer(timer)

        # total_loss = normal_loss * 0.5 + similarity_loss * 0.5
        similarity_loss = torch.mean(torch.stack(similarity_losses))
        total_loss = normal_loss * self.loss_weights[0] + similarity_loss * self.loss_weights[1]

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                    cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance, v_up = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            # total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            # center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            # up_point = center_point_c + total_up[:, 0]
            #
            # center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
            #                  :3].cpu().numpy()
            # up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
            #                                :3].cpu().numpy() - center_point_w)
            #
            arrows = o3d.geometry.TriangleMesh()
            # for i in range(center_point_w.shape[0]):
            #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
            #                                                    cylinder_height=0.00025, cone_height=0.00025,
            #                                                    resolution=3, cylinder_split=1)
            #     arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
            #     arrow.translate(center_point_w[i])
            #     arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using constrained up vector
class LModel12(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel12, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        v_up = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up = vector_to_sphere_coordinate(self.v_up)
        self.v_up[:, 0] = self.v_up[:, 0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up[:, 1] = self.v_up[:, 1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)
        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        v_up11 = (self.v_up[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        return seg_distance, v_up

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        seg_distance, v_up = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        edge_up_c = v_up[point_index].reshape((-1, 4, 3))
        edge_points = point_pos_c[point_index].reshape((-1, 4, 3))
        time_profile[0], timer = refresh_timer(timer)

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        similarity_losses = []
        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)
        similarity_losses.append(similarity_loss)

        # 8: Normal loss

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2
        time_profile[8], timer = refresh_timer(timer)

        # total_loss = normal_loss * 0.5 + similarity_loss * 0.5
        similarity_loss = torch.mean(torch.stack(similarity_losses))
        total_loss = normal_loss * self.loss_weights[0] + similarity_loss * self.loss_weights[1]

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                    cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance, v_up = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            # total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            # center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            # up_point = center_point_c + total_up[:, 0]
            #
            # center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
            #                  :3].cpu().numpy()
            # up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
            #                                :3].cpu().numpy() - center_point_w)
            #
            arrows = o3d.geometry.TriangleMesh()
            # for i in range(center_point_w.shape[0]):
            #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
            #                                                    cylinder_height=0.00025, cone_height=0.00025,
            #                                                    resolution=3, cylinder_split=1)
            #     arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
            #     arrow.translate(center_point_w[i])
            #     arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        # o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using constrained up vector and gaussian distribution
class LModel13(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method):
        super(LModel13, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        v_up = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)
        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=True)

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up = vector_to_sphere_coordinate(self.v_up)
        self.v_up[:, 0] = self.v_up[:, 0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up[:, 1] = self.v_up[:, 1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        self.loss_weights = v_weights
        self.img_method = v_img_method

    def denormalize(self, v_num_repeat):
        eps = torch.tensor(np.random.normal(0, 1, (v_num_repeat, self.gaussian_mean.shape[0])), dtype=torch.float32).to(
            self.gaussian_mean.device)
        eps = torch.clamp_min(eps, 1e-4)
        seg_distance = (self.gaussian_mean[None, :] + eps * self.gaussian_std[None, :]) * self.seg_distance_normalizer

        v_up11 = (self.v_up[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        return seg_distance, v_up

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = 10
        seg_distance, v_up = self.denormalize(num_repeat)
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        target_seg_distance = seg_distance[:, point_index].reshape(-1)
        point_index = point_index * seg_distance.shape[0]
        ray_c = self.ray_c[point_index]
        point_pos_c = ray_c * target_seg_distance[:, None]
        edge_points = point_pos_c.reshape((-1, 4, 3))
        edge_up_c = v_up[point_index].reshape((-1, 4, 3))
        time_profile[0], timer = refresh_timer(timer)

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        similarity_losses = []
        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        # 8: Normal loss
        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2
        time_profile[8], timer = refresh_timer(timer)

        # 9. Cannot lie on the observing plane
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.17 * torch.ones_like(cur_up[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.17)

        # total_loss = normal_loss * 0.5 + similarity_loss * 0.5
        total_loss = normal_loss * self.loss_weights[0] + similarity_loss * self.loss_weights[1] + \
                                          normalization_loss * self.loss_weights[2]

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                    cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, [normal_loss, similarity_loss, normalization_loss]

    def debug_save(self, v_index):
        seg_distance, v_up = self.denormalize(1)
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(r"output/img_field_test/imgs_log/target_{}_line.obj".format(v_index), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

    def len(self):
        return len(self.graph1.graph["faces"])

# Normal loss and Similarity loss using calculated up vector and gaussian distribution
class LModel14(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel14, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        v_up = []
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)
        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        # self.gaussian_std = nn.Parameter(torch.log(self.gaussian[:, 1]), requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=False)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root
        self.loss_weights = v_weights
        self.img_method = v_img_method

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0,0,1),),device=device,dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long),
                                     2, 1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long),
                                   2, 1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up_c[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_polygon(self, edge_points, v_dis, id_epoch):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point

        # Sample points on edges
        length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        num_per_edge_m2 = 10
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 500)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)

        # Sample points within triangle
        num_per_half_m2 = 100
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir)+1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 1000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1]+1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3]+1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3]+1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        # num_repeat = 100
        # id_end = num_edge_points.cumsum(0)
        # id_start = num_edge_points.cumsum(0)
        # id_start = id_start.roll(1)
        # id_start[0]=0
        # loss = []
        # for id_repeat in range(num_repeat):
        #     id_edge = 0 + id_repeat * 9
        #     img1 = torch.cat((
        #         sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs1[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     img2 = torch.cat((
        #         sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs2[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     loss.append((img1-img2).mean())

        # Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        # Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        is_debug=False
        if is_debug:
            id_edge_base = 0
            for id_repeat in range(100):
                id_edge = id_edge_base + id_repeat * 9
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_repeat, id_epoch)),
                            np.concatenate((img1, img2), axis=0))

                img1 = torch.cat((
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs1[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                img2 = torch.cat((
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                print("{}_{:.3f}_{:.1f}_{:.1f}".format(id_repeat, (img1-img2).mean(),
                      v_dis.reshape(-1,4)[id_edge,0].item(),
                      v_dis.reshape(-1,4)[id_edge,1].item(),)
                      )
            pass

        # 9: Viz
        if False and self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"output/img_field_test/imgs_log/2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(r"output/img_field_test/imgs_log/3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()

        return similarity_loss, normal_loss, normalization_loss

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = 1 if is_log else 100
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        repeat_point_index = point_index * num_repeat
        ray_c = self.ray_c[repeat_point_index].reshape((-1, 4, 3))

        # Target
        target_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,0:1]
        eps = _standard_normal(target_dis.shape, target_dis.dtype, target_dis.device)
        target_dis = (target_dis + self.gaussian_std[repeat_point_index].reshape(-1,4)[:,0:1] * eps) * self.seg_distance_normalizer
        # Reference
        ref_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,1:].detach() * self.seg_distance_normalizer
        seg_distance = torch.cat((target_dis,ref_dis), dim=1)
        point_pos_c = ray_c * seg_distance[:, :, None]
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, seg_distance, v_id_epoch)

        losses = torch.stack(losses).reshape((-1,3))

        id_target = 0
        total_loss = losses[id_target, 0] * self.loss_weights[0] + \
                     losses[id_target, 1] * self.loss_weights[1] + \
                     losses[id_target, 2] * self.loss_weights[2]
        # total_loss = losses[id_target, 0]
        time_profile[9], timer = refresh_timer(timer)

        if is_log:
            # Log 2D
            img1_base = self.rgb1.copy()
            shape = img1_base.shape[:2][::-1]
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            img1_base = cv2.polylines(img1_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, losses[id_target]

    def debug_save(self, v_index):
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_normal = normalize_tensor(torch.cross(total_edge_points[:, 1] - total_edge_points[:, 0], \
                            total_edge_points[:, 3] - total_edge_points[:, 1]))
            sign_flag = torch.sum(total_normal * torch.tensor(((0, 0, 1),),
                                                              device=point_pos_c.device, dtype=torch.float32),dim=1) > 0
            total_normal[sign_flag] = -total_normal[sign_flag]
            total_up = torch.cross(
                total_normal,
                total_edge_points[:, 1] - total_edge_points[:, 0]
            )

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,:3]\
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index)), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1,4))[:,0].mean()

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using calculated up vector and gaussian distribution
class LModel15(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root, v_num_gaussian_sample):
        super(LModel15, self).__init__()
        self.num_gaussian_sample = v_num_gaussian_sample
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        v_up = []
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)
        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        # self.gaussian_std = nn.Parameter(torch.log(self.gaussian[:, 1]), requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root
        self.loss_weights = v_weights
        self.img_method = v_img_method

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0,0,1),),device=device,dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long),
                                     2, 1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long),
                                   2, 1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up_c[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_edge(self, num_per_edge_m2, cur_dir, start_point):
        length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 100)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)
        return num_edge_points, sampled_edge_points

    def sample_polygon(self, num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point):
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir) + 1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 500)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)
        _t1 = torch.sqrt(sample_points1[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)
        return num_polygon_points, sampled_polygon_points

    def sample_points_based_on_polygon(self, edge_points, v_num_repeat, v_prob, id_epoch, v_is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0. Unpack data
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point
        time_profile[0], timer = refresh_timer(timer)

        # 1. Sample points on edges
        num_per_edge_m = 100
        num_edge_points, sampled_edge_points = self.sample_edge(num_per_edge_m, cur_dir, start_point)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample points within triangle
        # num_per_half_m2 = 50
        # num_polygon_points, sampled_polygon_points = self.sample_polygon(num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point)
        # time_profile[2], timer = refresh_timer(timer)

        # 3. Calculate pixel coordinate
        # coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)
        coordinates = sampled_edge_points

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3]+1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[3], timer = refresh_timer(timer)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[4], timer = refresh_timer(timer)

        # 5. Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3]+1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        time_profile[6], timer = refresh_timer(timer)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        # 7. Similarity loss
        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2, reduction='none').mean(dim=1)

        total_edge_points = num_edge_points.sum()
        edge_index = torch.arange(num_edge_points.shape[0], device=device).repeat_interleave(num_edge_points)
        # face_index = torch.arange(num_polygon_points.shape[0], device=device).repeat_interleave(num_polygon_points)
        edge_similarity = similarity_loss[:total_edge_points]
        # face_similarity = similarity_loss[total_edge_points:]
        edge_similarity = scatter_add(edge_similarity,edge_index,dim=0) / num_edge_points
        # face_similarity = scatter_add(face_similarity,face_index,dim=0) / num_polygon_points

        def spearmanr(pred, target, **kw):
            pred = torchsort.soft_rank(pred, **kw)
            target = torchsort.soft_rank(target, **kw)
            pred = pred - pred.mean()
            pred = pred / pred.norm()
            target = target - target.mean()
            target = target / target.norm()
            return (pred * target).sum()

        # total_similarity = (edge_similarity+face_similarity).reshape(-1, v_num_repeat) / 2
        total_similarity = edge_similarity.reshape(-1, v_num_repeat)
        v_prob = v_prob.reshape(-1,v_num_repeat)
        # similarity_loss = F.mse_loss(v_prob, 1 - total_similarity)
        similarity_loss = 1 - spearmanr(v_prob, 1 - total_similarity)

        time_profile[7], timer = refresh_timer(timer)

        # num_repeat = 100
        # id_end = num_edge_points.cumsum(0)
        # id_start = num_edge_points.cumsum(0)
        # id_start = id_start.roll(1)
        # id_start[0]=0
        # loss = []
        # for id_repeat in range(num_repeat):
        #     id_edge = 0 + id_repeat * 9
        #     img1 = torch.cat((
        #         sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs1[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     img2 = torch.cat((
        #         sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs2[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     loss.append((img1-img2).mean())

        # 8. Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        time_profile[8], timer = refresh_timer(timer)

        # 9. Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)
        time_profile[9], timer = refresh_timer(timer)

        is_debug=False
        if is_debug:
            for id_edge in range(self.num_gaussian_sample):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    # roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    # roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            np.concatenate((img1, img2), axis=0))
                sampled_imgs = (torch.stack([
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                ], dim = 0).detach().cpu().numpy()*255).astype(np.uint8).clip(0,255)
                cv2.imwrite(os.path.join(self.log_root, "s_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            sampled_imgs)
            pass

        # 9: Viz
        if False and v_is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"output/img_field_test/imgs_log/2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(r"output/img_field_test/imgs_log/3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))


        return similarity_loss, normal_loss, normalization_loss

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = self.num_gaussian_sample
        # num_repeat = 1 if is_log else 100
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        point_index = np.asarray(point_index).reshape(-1,4)
        point_index = point_index[5:6]
        repeat_point_index = np.tile(point_index[:,None,:], [1, num_repeat, 1],).flatten().tolist()
        ray_c = self.ray_c[repeat_point_index].reshape((-1, 4, 3))

        self.gaussian_mean.data = torch.clamp(self.gaussian_mean.data, 1 / self.seg_distance_normalizer, 500 / self.seg_distance_normalizer)
        self.gaussian_std.data = torch.clamp(self.gaussian_std.data, 1 / self.seg_distance_normalizer, 500 / self.seg_distance_normalizer)

        is_split = True
        if is_split:
            # Target
            target_mean = self.gaussian_mean[repeat_point_index].reshape(-1, 4)[:, 0:1]
            target_std = self.gaussian_std[repeat_point_index].reshape(-1, 4)[:, 0:1]
            dis = torch.distributions.Normal(target_mean[:, 0], target_std[:, 0])
            target_dis = dis.rsample()
            target_dis[0] = target_mean[0, 0]
            target_dis = torch.clamp(target_dis, 1 / self.seg_distance_normalizer, 500 / self.seg_distance_normalizer)
            prob = torch.exp(dis.log_prob(target_dis))
            normalized_prob = prob / torch.exp(dis.log_prob(target_mean[:, 0])).detach()
            target_dis = target_dis * self.seg_distance_normalizer
            # Reference
            ref_dis = self.gaussian_mean[repeat_point_index].reshape(-1, 4)[:,
                      1:].detach() * self.seg_distance_normalizer
            seg_distance = torch.cat((target_dis[:, None], ref_dis), dim=1)
            point_pos_c = ray_c * seg_distance[:, :, None]
            edge_points = point_pos_c
        else:
            target_mean = self.gaussian_mean[repeat_point_index]
            target_std = self.gaussian_std[repeat_point_index]
            dis = torch.distributions.Normal(target_mean, target_std)
            target_dis = dis.rsample()
            target_dis = torch.clamp(target_dis, 1 / self.seg_distance_normalizer, 500 / self.seg_distance_normalizer)
            prob = torch.exp(dis.log_prob(target_dis))
            normalized_prob = (prob / torch.exp(dis.log_prob(target_mean)).detach()).reshape(-1,4)[:,0]
            seg_distance = target_dis.reshape(-1, 4) * self.seg_distance_normalizer
            point_pos_c = ray_c * seg_distance[:, :, None]
            edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, num_repeat, normalized_prob, v_id_epoch, is_log)

        losses = torch.stack(losses).reshape((-1,3))

        id_target = 0
        total_loss = losses[id_target, 0] * self.loss_weights[0] + \
                     losses[id_target, 1] * self.loss_weights[1] + \
                     losses[id_target, 2] * self.loss_weights[2]
        # total_loss = losses[id_target, 0]
        time_profile[9], timer = refresh_timer(timer)

        if is_log:
            # Log 2D
            img1_base = self.rgb1.copy()
            shape = img1_base.shape[:2][::-1]
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            img1_base = cv2.polylines(img1_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, losses[id_target]

    def debug_save(self, v_index):
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_normal = normalize_tensor(torch.cross(total_edge_points[:, 1] - total_edge_points[:, 0], \
                            total_edge_points[:, 3] - total_edge_points[:, 1]))
            sign_flag = torch.sum(total_normal * torch.tensor(((0, 0, 1),),
                                                              device=point_pos_c.device, dtype=torch.float32),dim=1) > 0
            total_normal[sign_flag] = -total_normal[sign_flag]
            total_up = torch.cross(
                total_normal,
                total_edge_points[:, 1] - total_edge_points[:, 0]
            )

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,:3]\
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index)), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1,4))[:,0].mean()

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using calculated up vector without gaussian distribution
class LModel16(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel16, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        v_up = []
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.seg_distance = torch.tensor(
            [self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
            dtype=torch.float32) / self.seg_distance_normalizer
        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)
        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root
        self.loss_weights = v_weights
        self.img_method = v_img_method

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0,0,1),),device=device,dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long),
                                     2, 1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long),
                                   2, 1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up_c[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_edge(self, num_per_edge_m2, cur_dir, start_point):
        length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 2000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)
        return num_edge_points, sampled_edge_points

    def sample_polygon(self, num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point):
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir) + 1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 500)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)
        _t1 = torch.sqrt(sample_points1[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)
        return num_polygon_points, sampled_polygon_points

    def sample_points_based_on_polygon(self, edge_points, id_epoch, v_is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0. Unpack data
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point
        time_profile[0], timer = refresh_timer(timer)

        # 1. Sample points on edges
        num_per_edge_m = 100
        num_edge_points, sampled_edge_points = self.sample_edge(num_per_edge_m, cur_dir, start_point)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample points within triangle
        # num_per_half_m2 = 50
        # num_polygon_points, sampled_polygon_points = self.sample_polygon(num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point)
        # time_profile[2], timer = refresh_timer(timer)

        # 3. Calculate pixel coordinate
        # coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)
        coordinates = sampled_edge_points

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3]+1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[3], timer = refresh_timer(timer)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[4], timer = refresh_timer(timer)

        # 5. Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3]+1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        time_profile[6], timer = refresh_timer(timer)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        # 7. Similarity loss
        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        # num_repeat = 100
        # id_end = num_edge_points.cumsum(0)
        # id_start = num_edge_points.cumsum(0)
        # id_start = id_start.roll(1)
        # id_start[0]=0
        # loss = []
        # for id_repeat in range(num_repeat):
        #     id_edge = 0 + id_repeat * 9
        #     img1 = torch.cat((
        #         sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs1[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     img2 = torch.cat((
        #         sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs2[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     loss.append((img1-img2).mean())

        # 8. Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        time_profile[8], timer = refresh_timer(timer)

        # 9. Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)
        time_profile[9], timer = refresh_timer(timer)

        is_debug=False
        if is_debug:
            for id_edge in range(1):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    # roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    # roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            np.concatenate((img1, img2), axis=0))
                sampled_imgs = (torch.stack([
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                ], dim = 0).detach().cpu().numpy()*255).astype(np.uint8).clip(0,255)
                cv2.imwrite(os.path.join(self.log_root, "s_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            sampled_imgs)
            pass

        # 9: Viz
        if False and v_is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"output/img_field_test/imgs_log/2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(r"output/img_field_test/imgs_log/3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
        return similarity_loss, normal_loss, normalization_loss

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        # point_index = np.asarray(point_index).reshape(-1,4)
        # point_index = point_index[3:4]
        ray_c = self.ray_c[point_index].reshape((-1, 4, 3))
        seg_distance = self.seg_distance[point_index].reshape((-1, 4, 1)) * self.seg_distance_normalizer
        seg_distance[:,1:] = seg_distance[:,1:].detach()
        point_pos_c = ray_c * seg_distance
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, v_id_epoch, is_log)

        losses = torch.stack(losses).reshape((-1,3))

        id_target = 0
        total_loss = losses[id_target, 0] * self.loss_weights[0] + \
                     losses[id_target, 1] * self.loss_weights[1] + \
                     losses[id_target, 2] * self.loss_weights[2]
        # total_loss = losses[id_target, 0]
        time_profile[9], timer = refresh_timer(timer)

        if is_log:
            # Log 2D
            img1_base = self.rgb1.copy()
            shape = img1_base.shape[:2][::-1]
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            img1_base = cv2.polylines(img1_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, losses[id_target]

    def debug_save(self, v_index):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_normal = normalize_tensor(torch.cross(total_edge_points[:, 1] - total_edge_points[:, 0], \
                            total_edge_points[:, 3] - total_edge_points[:, 1]))
            sign_flag = torch.sum(total_normal * torch.tensor(((0, 0, 1),),
                                                              device=point_pos_c.device, dtype=torch.float32),dim=1) > 0
            total_normal[sign_flag] = -total_normal[sign_flag]
            total_up = torch.cross(
                total_normal,
                total_edge_points[:, 1] - total_edge_points[:, 0]
            )

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,:3]\
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index)), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

        return 0

    def len(self):
        return len(self.graph1.graph["faces"])