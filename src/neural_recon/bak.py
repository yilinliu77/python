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
                                             0.17 * torch.ones_like(cur_up[:, 0]))  # [80, 90] degrees
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
            colors[:, 0] = 1
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

        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
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
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
                                             0.5 * torch.ones_like(cur_up[:, 0]))  # [60, 90] degrees
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
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir) + 1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 1000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3] + 1e-6)
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
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3] + 1e-6)
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

        is_debug = False
        if is_debug:
            id_edge_base = 0
            for id_repeat in range(100):
                id_edge = id_edge_base + id_repeat * 9
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    roi_coor_2d[num_edge_points.sum() + num_polygon_points[
                                                        :id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                :id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[
                                                             :id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                     :id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_repeat, id_epoch)),
                            np.concatenate((img1, img2), axis=0))

                img1 = torch.cat((
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs1[num_edge_points.sum() + num_polygon_points[
                                                         :id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                 :id_edge + 1].sum()],
                ), dim=0)
                img2 = torch.cat((
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points.sum() + num_polygon_points[
                                                         :id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                 :id_edge + 1].sum()],
                ), dim=0)
                print("{}_{:.3f}_{:.1f}_{:.1f}".format(id_repeat, (img1 - img2).mean(),
                                                       v_dis.reshape(-1, 4)[id_edge, 0].item(),
                                                       v_dis.reshape(-1, 4)[id_edge, 1].item(), )
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
        target_dis = self.gaussian_mean[repeat_point_index].reshape(-1, 4)[:, 0:1]
        eps = _standard_normal(target_dis.shape, target_dis.dtype, target_dis.device)
        target_dis = (target_dis + self.gaussian_std[repeat_point_index].reshape(-1, 4)[:,
                                   0:1] * eps) * self.seg_distance_normalizer
        # Reference
        ref_dis = self.gaussian_mean[repeat_point_index].reshape(-1, 4)[:, 1:].detach() * self.seg_distance_normalizer
        seg_distance = torch.cat((target_dis, ref_dis), dim=1)
        point_pos_c = ray_c * seg_distance[:, :, None]
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, seg_distance, v_id_epoch)

        losses = torch.stack(losses).reshape((-1, 3))

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
                                                              device=point_pos_c.device, dtype=torch.float32),
                                  dim=1) > 0
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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
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

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1, 4))[:, 0].mean()

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

        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
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
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
                                             0.5 * torch.ones_like(cur_up[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_edge(self, num_per_edge_m2, cur_dir, start_point):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3] + 1e-6)
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
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3] + 1e-6)
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
        edge_similarity = scatter_add(edge_similarity, edge_index, dim=0) / num_edge_points

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
        v_prob = v_prob.reshape(-1, v_num_repeat)
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

        is_debug = False
        if is_debug:
            for id_edge in range(self.num_gaussian_sample):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
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
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                ], dim=0).detach().cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
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
        point_index = np.asarray(point_index).reshape(-1, 4)
        point_index = point_index[5:6]
        repeat_point_index = np.tile(point_index[:, None, :], [1, num_repeat, 1], ).flatten().tolist()
        ray_c = self.ray_c[repeat_point_index].reshape((-1, 4, 3))

        self.gaussian_mean.data = torch.clamp(self.gaussian_mean.data, 1 / self.seg_distance_normalizer,
                                              500 / self.seg_distance_normalizer)
        self.gaussian_std.data = torch.clamp(self.gaussian_std.data, 1 / self.seg_distance_normalizer,
                                             500 / self.seg_distance_normalizer)

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
            normalized_prob = (prob / torch.exp(dis.log_prob(target_mean)).detach()).reshape(-1, 4)[:, 0]
            seg_distance = target_dis.reshape(-1, 4) * self.seg_distance_normalizer
            point_pos_c = ray_c * seg_distance[:, :, None]
            edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, num_repeat, normalized_prob, v_id_epoch, is_log)

        losses = torch.stack(losses).reshape((-1, 3))

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
                                                              device=point_pos_c.device, dtype=torch.float32),
                                  dim=1) > 0
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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
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

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1, 4))[:, 0].mean()

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

        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
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
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
                                             0.5 * torch.ones_like(cur_up[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_edge(self, num_per_edge_m2, cur_dir, start_point):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
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
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3] + 1e-6)
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
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3] + 1e-6)
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

        is_debug = False
        if is_debug:
            for id_edge in range(1):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
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
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                ], dim=0).detach().cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
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
        seg_distance[:, 1:] = seg_distance[:, 1:].detach()
        point_pos_c = ray_c * seg_distance
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, v_id_epoch, is_log)

        losses = torch.stack(losses).reshape((-1, 3))

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
                                                              device=point_pos_c.device, dtype=torch.float32),
                                  dim=1) > 0
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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
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


# Normal loss and Similarity loss using calculated up vector without gaussian distribution
class LModel17(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel17, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method

        # Graph related
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]

        # Buffer variables
        ray_c_vertex = torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32)
        ray_c_centroid = torch.tensor(self.graph1.graph["face_center"]["ray_c"], dtype=torch.float32)
        self.register_buffer("ray_c", torch.cat((ray_c_vertex, ray_c_centroid), dim=0))  # (M, 2)
        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field1", torch.asarray(v_data["edge_field1"]).permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field2", torch.asarray(v_data["edge_field2"]).permute(2, 0, 1).unsqueeze(0))

        # Image models
        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        # for p in self.img_model2.parameters():
        #     p.requires_grad = False

        # Edge index
        # Start, end, prev, next
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

        # Trained parameters
        # Distance parameters
        # Normalized to [0,1]
        self.seg_distance_normalizer = 300
        seg_distance_vertex = torch.tensor([
            self.graph1.nodes[id_node]["distance"].tolist() for id_node in self.graph1.nodes()
        ], dtype=torch.float32) / self.seg_distance_normalizer
        seg_distance_centroid = torch.tensor(self.graph1.graph["face_center"]["distance"],
                                             dtype=torch.float32) / self.seg_distance_normalizer
        self.seg_distance = nn.Parameter(torch.cat((seg_distance_vertex, seg_distance_centroid), dim=0),
                                         requires_grad=True)
        self.id_centroid_start = seg_distance_vertex.shape[0]
        # Up vector
        # Method 1: Define a 3-dimensional vector
        if False:
            v_up = []
            self.v_up_dict = {}
            for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
                for id_segment in range(len(face_ids)):
                    id_start = face_ids[id_segment]
                    id_end = face_ids[(id_segment + 1) % len(face_ids)]
                    if id_start > id_end:
                        t = id_start
                        id_start = id_end
                        id_end = t
                    self.v_up_dict[(id_start, id_end, id_patch)] = len(v_up)
                    v_up.append(
                        torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32))
            self.v_up = torch.stack(v_up, dim=0)

        # Method 2: Define a 1 parameter up vector which is normalized by 2*pi
        num_edges_per_face = [len(item) for item in self.graph1.graph["faces"]]
        face_to_up = torch.tensor(np.insert(np.cumsum(num_edges_per_face), 0, 0), dtype=torch.long)
        self.register_buffer("face_to_up", face_to_up)

        self.edge_to_up = {}
        v_up = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                if id_start > id_end:
                    t = id_start
                    id_start = id_end
                    id_end = t
                self.edge_to_up[(id_start, id_end, id_patch)] = len(v_up)
                v_up.append(torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32))

        all_edge_points = (self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer)[
            list(itertools.chain(*self.edge_point_index))].reshape(-1, 4, 3)
        start_points, end_points = all_edge_points[:, 0], all_edge_points[:, 1]
        t_candidate = torch.arange(0, 1, 0.1)
        start_points = start_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        end_points = end_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        all_t_candidate = t_candidate.tile(all_edge_points.shape[0]) * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(all_t_candidate) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            a * torch.cos(all_t_candidate) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate)
        ), dim=1)
        up_c = normalize_tensor(up_c)
        up_c = up_c.reshape(-1, t_candidate.shape[0], 3)
        distance = (up_c * torch.stack(v_up, dim=0)[:, None, :]).sum(dim=-1)
        id_best = distance.argmax(dim=1)
        self.v_up = t_candidate[id_best]

        # Register parameters
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        # Visualization
        viz_shape = (6000, 4000)
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root

        # Accurate initialization in patch 1522
        id_vertices = np.asarray(self.edge_point_index[1522]).reshape(-1, 4)[:, 0]
        self.seg_distance.data[id_vertices] = torch.tensor(
            [0.3040, 0.3033, 0.3030, 0.3021, 0.3110, 0.3067, 0.3063, 0.3057, 0.3045])
        # self.seg_distance.data[id_vertices] = torch.tensor([0.3040,0.3033,0.3030,0.3021,0.3115,0.3067,0.3063,0.3057,0.3045])
        self.seg_distance.data[self.id_centroid_start + 1522] = 0.3030

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
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
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c, v_id_epoch, is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))
        cur_up = normalize_tensor(edge_up_c)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.2).to(device)  # m
        half_window_size_step = 0.01

        # Compute interpolated point
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
            cur_up.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        similarity_loss = self.compute_similarity(interpolated_coordinates_camera,
                                                  self.intrinsic1, self.intrinsic2, self.extrinsic1, self.extrinsic2)

        # Normal Loss
        # Input: start_point and cur_up
        p1 = start_point
        p2 = start_point + cur_up
        p = torch.cat([p1, p2], dim=0)
        A = torch.stack([p[:, 0], p[:, 1], torch.ones_like(p[:, 0])], dim=1)
        solution = torch.linalg.lstsq(A, p[:, 2:3])
        normal_loss = (((A * solution.solution[:, 0]).sum(axis=1, keepdims=True) - p[:,
                                                                                   2:3]) ** 2).mean() / self.seg_distance_normalizer
        # cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        # next_dir = edge_points[:, 3] - edge_points[:, 1]
        # next_up = torch.cross(edge_up_c[:, 1], next_dir)
        # next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        # prev_dir = edge_points[:, 0] - edge_points[:, 2]
        # prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        # prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        # normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        # normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        # normal_loss = (normal_loss1 + normal_loss2) / 2
        #
        # observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        # should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
        #                                      0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        # normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        if is_log and self.id_viz_patch != -1:
            line_thickness = 1
            point_thickness = 2
            point_radius = 1
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ \
                             torch.inverse(self.extrinsic1)

            polygon_points_2d_1 = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
            polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
            polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()

            for id_edge, _ in enumerate(num_coordinates_per_edge):
                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                polygon_2d1 = (self.intrinsic1 @ start_point.T).T
                polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
                polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.polylines(line_img1, [polygon_2d1],
                              isClosed=True, color=(0, 255, 0), thickness=line_thickness, lineType=cv2.LINE_AA)
                for point in polygon_2d1:
                    cv2.circle(line_img1, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = polygon_points_2d_1[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]]
                roi_coor_2d1_numpy = np.clip(roi_coor_2d1_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                # Image 2
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]

                polygon_2d2 = (transformation @ to_homogeneous_tensor(start_point).T).T
                polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
                polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.polylines(line_img2, [polygon_2d2],
                              isClosed=True, color=(0, 255, 0), thickness=line_thickness, lineType=cv2.LINE_AA)
                for point in polygon_2d2:
                    cv2.circle(line_img2, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d2_numpy = polygon_points_2d_2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]]
                roi_coor_2d2_numpy = np.clip(roi_coor_2d2_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))

        return similarity_loss

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
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

    def sample_triangles(self, num_per_m, p1, p2, p3, num_max_sample=500):
        d1 = p2 - p1
        d2 = p3 - p2
        area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

        num_edge_points, edge_points = self.sample_edge(num_per_m,
                                                        torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)

        num_per_m2 = num_per_m * num_per_m
        num_tri_samples = torch.clamp((area * num_per_m2).to(torch.long), 1, num_max_sample * 4)
        samples = torch.rand(num_tri_samples.sum(), 2).to(p1.device)
        _t1 = torch.sqrt(samples[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * p1.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * (1 - samples[:, 1:2]) * p2.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * samples[:, 1:2] * p3.repeat_interleave(num_tri_samples, dim=0)

        num_total_points = num_edge_points + num_tri_samples
        num_total_points_cumsum = num_total_points.cumsum(0).roll(1)
        num_total_points_cumsum[0] = 0
        sampled_total_points = torch.zeros((num_total_points.sum(), 3), device=p1.device, dtype=torch.float32)
        num_edge_points_ = num_edge_points.cumsum(0).roll(1)
        num_edge_points_[0] = 0
        num_tri_points_ = num_tri_samples.cumsum(0).roll(1)
        num_tri_points_[0] = 0
        edge_index = torch.arange(num_edge_points.sum(), device=p1.device) \
                     - num_edge_points_.repeat_interleave(num_edge_points) \
                     + num_total_points_cumsum.repeat_interleave(num_edge_points)
        tri_index = torch.arange(num_tri_samples.sum(), device=p1.device) \
                    - num_tri_points_.repeat_interleave(num_tri_samples) \
                    + num_total_points_cumsum.repeat_interleave(num_tri_samples) \
                    + num_edge_points.repeat_interleave(num_tri_samples)
        sampled_total_points[edge_index] = edge_points
        sampled_total_points[tri_index] = sampled_polygon_points
        return num_total_points, sampled_total_points

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
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3] + 1e-6)
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
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3] + 1e-6)
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

        is_debug = False
        if is_debug:
            for id_edge in range(1):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
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
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                ], dim=0).detach().cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
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

    def sample_points_based_on_convex(self, edge_points, centroid_c, v_id_epoch, v_is_log):
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
        num_per_m = 20
        num_polygon_points, sampled_polygon_points = self.sample_triangles(num_per_m,
                                                                           start_point, end_point, centroid_c,
                                                                           num_max_sample=500)
        time_profile[2], timer = refresh_timer(timer)

        # 3. Calculate pixel coordinate
        # coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)
        coordinates = sampled_edge_points
        num_coordinates = num_edge_points
        similarity_loss = self.compute_similarity(coordinates,
                                                  self.intrinsic1, self.intrinsic2, self.extrinsic1, self.extrinsic2)
        time_profile[3], timer = refresh_timer(timer)
        # 9: Viz
        if v_is_log and self.id_viz_patch != -1:
            line_thickness = 1
            point_thickness = 1
            point_radius = 2
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ \
                             torch.inverse(self.extrinsic1)

            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]
            line_img2 = self.rgb2.copy()

            polygon_points_2d_1 = (self.intrinsic1 @ coordinates.T).T
            polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
            polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_1 = (np.clip(polygon_points_2d_1, 0, 0.99999) * shape).astype(np.int32)
            polygon_points_2d_2 = (np.clip(polygon_points_2d_2, 0, 0.99999) * shape).astype(np.int32)

            polygon_2d1 = (self.intrinsic1 @ start_point.T).T
            polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
            polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon_2d1],
                          isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            for point in polygon_2d1:
                cv2.circle(line_img1, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

            # Image 2
            polygon_2d2 = (transformation @ to_homogeneous_tensor(start_point).T).T
            polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
            polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img2, [polygon_2d2],
                          isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            for point in polygon_2d2:
                cv2.circle(line_img2, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

            for id_edge in range(start_point.shape[0]):
                img1 = line_img1.copy()
                img2 = line_img2.copy()

                id_polygon_coord = num_coordinates[:id_edge].sum()
                for point in polygon_points_2d_1[id_polygon_coord:id_polygon_coord + num_coordinates[id_edge]]:
                    cv2.circle(img1, point, radius=point_radius, color=(0, 0, 255), thickness=point_thickness)

                id_polygon_coord = num_coordinates[:id_edge].sum()
                for point in polygon_points_2d_2[id_polygon_coord:id_polygon_coord + num_coordinates[id_edge]]:
                    cv2.circle(img2, point, radius=point_radius, color=(0, 0, 255), thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, v_id_epoch)),
                            np.concatenate((img1, img2), axis=0))

        return similarity_loss

    def sample_points_2d_(self, edge_points_c, v_intrinsic, v_is_log=False, v_img=None):
        device = edge_points_c.device
        edge_points_2d = (v_intrinsic @ edge_points_c.reshape(-1, 3).T).T.reshape(edge_points_c.shape)
        edge_points_2d = edge_points_2d[:, :, :2] / edge_points_2d[:, :, 2:]
        cur_dir_2d = edge_points_2d[:, 1] - edge_points_2d[:, 0]
        z_vector = to_homogeneous_tensor(torch.zeros_like(cur_dir_2d))  # The z axis of the image
        up_2d = normalize_tensor(
            torch.cross(to_homogeneous_tensor(cur_dir_2d.reshape(-1, 2)), z_vector)[:, :2].reshape(cur_dir_2d.shape))

        length = torch.linalg.norm(cur_dir_2d + 1e-6, dim=1)
        num_horizontal = 100
        num_vertical = 20
        horizontal_step = length / num_horizontal
        vertical_step = 1 / 4000
        num_coordinates_per_edge = torch.ones_like(edge_points_2d[:, 0, 0],
                                                   dtype=torch.long) * num_horizontal * num_vertical

        dx = (torch.arange(num_horizontal).to(device) / (num_horizontal - 1 + 1e-6) * length[:, None]).reshape(-1)
        dy = (torch.arange(num_vertical) - num_vertical / 2).to(device) / (
                num_vertical - 1 + 1e-6) * vertical_step * num_vertical

        # Meshgrid
        total_num_coords = dx.shape[0] * dy.shape[0]
        coords_x = dx.repeat_interleave(dy.shape[0])  # (total_num_coords,)
        coords_y = torch.tile(dy, (dx.shape[0],))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)

        normalized_cur_dir_2d = normalize_tensor(cur_dir_2d)
        roi_coor_2d = \
            normalized_cur_dir_2d.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            up_2d.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            (edge_points_2d[:, 0].repeat_interleave(num_coordinates_per_edge, dim=0))

        viz_data = []
        # 9: Viz
        if v_is_log:
            for id_edge, _ in enumerate(num_coordinates_per_edge):
                line_img1 = v_img.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d1_numpy = np.clip(roi_coor_2d1_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                viz_data.append(line_img1)

        return roi_coor_2d, viz_data

    def sample_points_2d(self, edge_points, id_epoch, v_is_log):
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

        # 1. Calculate projected edges and sample points within rectangle
        roi_coor_2d1, viz_data1 = self.sample_points_2d_(edge_points, self.intrinsic1, v_is_log, self.rgb1)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample pixel color
        roi_coor_2d1 = torch.clamp(roi_coor_2d1, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d1[None, :, :])[0]
        time_profile[2], timer = refresh_timer(timer)

        # 3. Second img
        edge_points_img2 = (self.extrinsic2 @ torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(
            edge_points.reshape(-1, 3)).T).T[:, :3].reshape(edge_points.shape)
        roi_coor_2d2, viz_data2 = self.sample_points_2d_(edge_points_img2, self.intrinsic2, v_is_log, self.rgb2)
        roi_coor_2d2 = torch.clamp(roi_coor_2d2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d2[None, :, :])[0]
        time_profile[3], timer = refresh_timer(timer)

        # 4. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2)
        time_profile[4], timer = refresh_timer(timer)

        # 5. viz
        if v_is_log:
            for idx in range(len(viz_data1)):
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(idx, id_epoch)),
                            np.concatenate((viz_data1[idx], viz_data2[idx]), axis=0))
        return similarity_loss, similarity_loss, similarity_loss

    def get_up_vector1(self, v_face_index):
        up_index = list(itertools.chain(
            *[np.insert(np.asarray(self.edge_point_index[face_id]).reshape(-1, 4)[:, :2], 2, face_id.cpu().item(),
                        axis=1)
              for face_id in v_face_index]))
        up_index = np.asarray(up_index)
        # Make sure the x index is lower than y
        up_index[up_index[:, 0] > up_index[:, 1]] = up_index[up_index[:, 0] > up_index[:, 1]][:, [1, 0, 2]]
        up_index = [self.v_up_dict[tuple(key.tolist())] for key in up_index]
        return up_index

    def get_up_vector2(self, v_face_index, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t = torch.cat([self.v_up[self.face_to_up[id_face]:self.face_to_up[id_face + 1]] for id_face in v_face_index],
                      dim=0)
        t = t * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            torch.sqrt(a * a + b * b) * torch.sin(t)
        ), dim=1)
        return normalize_tensor(up_c)

    def compute_similarity(self, coords_c,
                           v_intrinsic1, v_intrinsic2, v_extrinsic1, v_extrinsic2):
        coods2d_1 = (v_intrinsic1 @ coords_c.T).T
        coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        valid_mask1 = torch.logical_and(coods2d_1 > 0, coods2d_1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, coods2d_1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, coods2d_1[None, :, :])[0]

        # 5. Second img
        transformation = to_homogeneous_mat_tensor(v_intrinsic2) @ v_extrinsic2 @ torch.inverse(v_extrinsic1)
        coods2d_2 = (transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, coods2d_2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, coods2d_2[None, :, :])[0]

        # 7. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2)

        return similarity_loss

    def compute_edge_response(self, v_start_point, v_end_point):
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        num_edge_points, coords_c = self.sample_edge(100, v_end_point - v_start_point, v_start_point, 2000)
        # First image
        # start_2d1 = (self.intrinsic1 @ v_start_point.T).T
        # start_2d1 = start_2d1[:,:2] / start_2d1[:,2:3]
        # end_2d1 = (self.intrinsic1 @ v_end_point.T).T
        # end_2d1 = end_2d1[:,:2] / end_2d1[:,2:3]
        #
        # coods2d_1 = (self.intrinsic1 @ coords_c.T).T
        # coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        # coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # # Sample edge response
        # coordinate_tensor = coods2d_1.unsqueeze(0).unsqueeze(0)
        # sampled_edge1 = torch.nn.functional.grid_sample(self.edge_field1, coordinate_tensor * 2 - 1,
        #                                               align_corners=True)[0]  # [0,1] -> [-1,1]
        # dir_2d1 = normalize_tensor(end_2d1 - start_2d1)
        # edge_direction1 = normalize_tensor(sampled_edge1[0:2,0].T)
        # edge_weights1 = torch.abs((edge_direction1 * dir_2d1.repeat_interleave(num_edge_points,dim=0)).sum(dim=1))
        # edge_magnitude1 = sampled_edge1[2,0]

        # 5. Second img
        start_2d2 = (transformation @ to_homogeneous_tensor(v_start_point).T).T
        start_2d2 = start_2d2[:, :2] / start_2d2[:, 2:3]
        end_2d2 = (transformation @ to_homogeneous_tensor(v_end_point).T).T
        end_2d2 = end_2d2[:, :2] / end_2d2[:, 2:3]

        coods2d_2 = (transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        coordinate_tensor = coods2d_2.unsqueeze(0).unsqueeze(0)
        sampled_edge2 = torch.nn.functional.grid_sample(self.edge_field2, coordinate_tensor * 2 - 1,
                                                        align_corners=True)[0]  # [0,1] -> [-1,1]

        dir_2d2 = normalize_tensor(end_2d2 - start_2d2)

        edge_direction2 = normalize_tensor(sampled_edge2[0:2, 0].T)
        edge_weights2 = 1 - torch.abs((edge_direction2 * dir_2d2.repeat_interleave(num_edge_points, dim=0)).sum(dim=1))
        edge_magnitude2 = sampled_edge2[2, 0]

        # debug_va = scatter_add(edge_weights2, torch.arange(num_edge_points.shape[0],device=v_start_point.device).repeat_interleave(num_edge_points))
        # edge_response = (edge_weights1 * edge_magnitude1 + edge_weights2 * edge_magnitude2).sum()
        if True:
            edge_response = (edge_weights2 * edge_magnitude2).sum()
        else:
            edge_response_ = edge_weights2 * edge_magnitude2
            edge_response = scatter_add(edge_response_, torch.arange(
                num_edge_points.shape[0], device=v_start_point.device).repeat_interleave(num_edge_points))
            edge_response = (edge_response / num_edge_points).mean()

        return edge_response

    def forward(self, v_face_index, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_face_index]))
        point_index = np.asarray(point_index).reshape(-1, 4)[4:5].tolist()
        ray_c = self.ray_c[point_index].reshape((-1, 4, 3))
        seg_distance = self.seg_distance[point_index].reshape((-1, 4, 1)) * self.seg_distance_normalizer
        edge_points = ray_c * seg_distance
        up_c = self.get_up_vector2(v_face_index, edge_points[:, 0], edge_points[:, 1])
        # edge_points[:,1:] = edge_points[:,1:].detach()
        # centroid
        id_centroid = torch.tensor(list(itertools.chain(*[
            [item + self.id_centroid_start] * (len(self.edge_point_index[item]) // 4) for item in v_face_index])),
                                   device=v_face_index.device)
        id_centroid = id_centroid[4:5]
        centroid_point_c = (self.seg_distance[id_centroid] * self.seg_distance_normalizer)[:, None] * \
                           self.ray_c[id_centroid]

        edge_loss = self.compute_edge_response(v_start_point=edge_points[:, 0], v_end_point=edge_points[:, 1])
        similarity_loss = self.sample_points_based_on_convex(edge_points, centroid_point_c, v_id_epoch, is_log)
        # similarity_loss = self.sample_points_based_on_up(edge_points, up_c, v_id_epoch, is_log)

        # losses = []
        # losses += self.sample_points_based_on_convex(edge_points, centroid_point_c, v_id_epoch, is_log)
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        # losses += self.sample_points_based_on_polygon(edge_points, v_id_epoch, is_log)
        # losses += self.sample_points_2d(edge_points, v_id_epoch, is_log)
        # losses += self.sample_points_based_on_up(edge_points, up_c, v_id_epoch, is_log)
        # losses = torch.stack(losses).reshape((-1,3))

        weighted_similarity = similarity_loss * 1
        weighted_edge = edge_loss * (-1 / 10000)
        # weighted_edge = edge_loss * (-1)
        total_loss = weighted_similarity

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
                                      thickness=1)
            for point in roi_2d_numpy:
                cv2.circle(img1_base, (point * shape).astype(np.int32), radius=1,
                           color=(0, 255, 255),
                           thickness=2)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=1)
            for point in roi_2d_numpy:
                cv2.circle(img2_base, (point * shape).astype(np.int32), radius=1,
                           color=(0, 255, 255),
                           thickness=2)
            cv2.imwrite(os.path.join(self.log_root, r"2d_{:05d}.jpg".format(v_id_epoch)),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, [weighted_similarity, weighted_edge, total_loss]

    def debug_save(self, v_index):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_points, v_up_c):
            total_edge_points = v_edge_points

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + v_up_c

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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # id_target = torch.tensor((119,),dtype=torch.long, device=point_pos_c.device)
        id_target = torch.tensor((1522,), dtype=torch.long, device=point_pos_c.device)
        # Visualize target patch
        edge_points = point_pos_c[self.edge_point_index[id_target]].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(id_target, edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c)
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index + 1)), arrows)
        id_points = np.asarray(self.edge_point_index[id_target]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index + 1)), start_point_w, edge_index)

        # Visualize whole patch
        # edge_points = point_pos_c[list(itertools.chain(*self.edge_point_index))].reshape(-1,4,3)
        # up_c = self.get_up_vector2(
        #     torch.arange(len(self.edge_point_index), device=self.ray_c.device), edge_points[:,0], edge_points[:,1])
        # arrows = get_arrow(edge_points, up_c)
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(v_index)), arrows)
        # start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
        #     .cpu().numpy()
        # edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(v_index)), start_point_w, edge_index)
        pass

        return 0

    def len(self):
        return len(self.graph1.graph["faces"])


class Single_face_dataset(torch.utils.data.Dataset):
    def __init__(self, v_edge_indexes, id_face, v_training_mode):
        super(Single_face_dataset, self).__init__()
        self.data = np.concatenate([np.asarray(item).reshape(-1, 4) for item in v_edge_indexes[id_face:id_face + 1]],
                                   axis=0)
        self.id_edge_start = np.cumsum([len(item) // 4 for item in v_edge_indexes])[id_face - 1]
        self.length = self.data.shape[0]
        self.training_mode = v_training_mode
        pass

    def __getitem__(self, index):
        return torch.tensor(self.id_edge_start + index, dtype=torch.long), torch.tensor(self.data[index],
                                                                                        dtype=torch.long)

    def __len__(self):
        return self.length


class Edge_dataset(torch.utils.data.Dataset):
    def __init__(self, v_edge_indexes, is_one_target, id_edge, v_training_mode):
        super(Edge_dataset, self).__init__()
        self.id_faces = torch.arange(len(v_edge_indexes)).repeat_interleave(
            torch.tensor([len(item) // 4 for item in v_edge_indexes]))
        self.data = np.concatenate([np.asarray(item).reshape(-1, 4) for item in v_edge_indexes], axis=0)

        self.is_one_target = is_one_target
        self.id_edge = id_edge
        if not self.is_one_target:
            self.length = self.data.shape[0]
        else:
            self.length = 1000
        self.training_mode = v_training_mode
        pass

    def __getitem__(self, index):
        if self.is_one_target or self.training_mode == "validation":
            return self.id_edge, torch.tensor(self.data[self.id_edge], dtype=torch.long)
        else:
            return index, torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        if self.training_mode == "training":
            return self.length
        else:
            return 1


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_length, v_id_target, v_training_mode):
        super(Dummy_dataset, self).__init__()
        self.length = v_length
        self.id_target = v_id_target
        self.training_mode = v_training_mode
        pass

    def __getitem__(self, index):
        if self.id_target != -1:
            return torch.tensor(self.id_target, dtype=torch.long)
        else:
            return torch.tensor(index, dtype=torch.long)

    def __len__(self):
        if self.id_target != -1:
            if self.training_mode == "training":
                return 1000
            else:
                return 1
        else:
            return self.length



class LModel18(nn.Module):
    def __init__(self, v_data, v_viz_patch, v_viz_edge, v_weights, v_img_method, v_log_root):
        super(LModel18, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method
        self.log_root = v_log_root

        self.init_regular_variables(v_data)
        self.register_variables(v_data)
        self.register_img_model(v_data)
        self.calculate_index()
        self.register_distances(True)
        self.register_up_vectors2(True)

        # Debug
        self.id_viz_face = v_viz_patch
        self.id_viz_edge = v_viz_edge

        # Accurate initialization in patch 1476
        id_vertices = np.asarray(self.batched_points_per_patch[1476]).reshape(-1, 4)[:, 0]
        # self.seg_distance.data[id_vertices] = torch.tensor(
        #     [0.3040, 0.3033, 0.3030, 0.3026, 0.3107, 0.3067, 0.3063, 0.3057, 0.3045])
        # self.seg_distance.data[id_vertices] = torch.tensor([0.3040,0.3033,0.3030,0.3021,0.3115,0.3067,0.3063,0.3057,0.3045])
        # self.seg_distance.data[self.id_centroid_start + 1522] = 0.3030

    # Init-related methods
    def init_regular_variables(self, v_data):
        # Graph related
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]

        # Visualization
        # viz_shape = (6000, 4000)
        self.rgb1 = v_data["rgb1"]
        self.rgb2 = v_data["rgb2"]
        # self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        # self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

    def register_variables(self, v_data):
        transformation = to_homogeneous(v_data["intrinsic2"]) @ v_data["extrinsic2"] \
                         @ np.linalg.inv(v_data["extrinsic1"])
        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("transformation", torch.as_tensor(transformation).float())
        self.register_buffer("edge_field1", torch.asarray(v_data["edge_field1"]).permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field2", torch.asarray(v_data["edge_field2"]).permute(2, 0, 1).unsqueeze(0))

    def register_img_model(self, v_data):
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        # Image models
        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

    def calculate_index(self):
        # Edge index
        # Start, end, prev, next
        self.batched_points_per_patch = [[] for _ in self.graph1.graph["faces"]]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.batched_points_per_patch[id_patch].append(id_start)
                self.batched_points_per_patch[id_patch].append(id_end)
                self.batched_points_per_patch[id_patch].append(id_prev)
                self.batched_points_per_patch[id_patch].append(id_next)

    def register_distances(self, train_distance=True):
        self.seg_distance_normalizer = 300
        distances_scale = torch.tensor([self.graph1.nodes[id_node]["scale"] for id_node in self.graph1.nodes()],
                                       dtype=torch.float32) / self.seg_distance_normalizer
        self.register_buffer("scale", distances_scale)

        ray_c_vertex = torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32)
        ray_c_centroid = torch.tensor(self.graph1.graph["ray_c"], dtype=torch.float32)
        self.register_buffer("ray_c", ray_c_vertex)
        self.register_buffer("center_ray_c", ray_c_centroid)

        seg_distance_vertex = torch.tensor([
            self.graph1.nodes[id_node]["distance"].tolist() for id_node in self.graph1.nodes()
        ], dtype=torch.float32) / self.seg_distance_normalizer
        self.seg_distance = nn.Parameter(seg_distance_vertex, requires_grad=train_distance)
        self.id_centroid_start = seg_distance_vertex.shape[0]

    def register_up_vectors1(self, train_up_vectors):
        num_total_edge = np.sum([len(face) for face in self.graph1.graph["faces"]])

        id_edge_to_id_face = torch.zeros((num_total_edge, 2), dtype=torch.long)
        self.id_edge_to_id_up_dict = {}
        id_edge_to_id_up = []
        v_up = []

        # These three variables are used to initialize the up vector
        all_edge_points = self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer
        start_points = []
        end_points = []
        id_total_seg = 0
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                id_edge_to_id_up.append([])
                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_start, id_end)])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    if len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 1:
                        v_up.append(-up_c)
                    else:
                        assert len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 2
                        for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                            if id_patch == key:
                                continue
                            v_up.append(torch.tensor(
                                self.graph1.edges[(id_end, id_start)]["up_c"][key], dtype=torch.float32))
                    start_points.append(all_edge_points[id_end])
                    end_points.append(all_edge_points[id_start])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_end, id_start)])

                id_edge_to_id_face[id_total_seg, 0] = id_patch
                key2 = -1
                for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                    if id_patch == key:
                        continue
                    key2 = key
                id_edge_to_id_face[id_total_seg, 1] = key2
                id_total_seg += 1

        self.register_buffer("id_edge_to_id_up", torch.tensor(id_edge_to_id_up))
        self.register_buffer("id_edge_to_id_face", id_edge_to_id_face)

        # Initialization
        start_points = torch.stack(start_points, dim=0)
        end_points = torch.stack(end_points, dim=0)
        t_candidate = torch.arange(0, 1, 0.1)
        start_points = start_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        end_points = end_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        all_t_candidate = t_candidate.tile(len(v_up)) * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(all_t_candidate) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            a * torch.cos(all_t_candidate) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate)
        ), dim=1)
        up_c = normalize_tensor(up_c)
        up_c = up_c.reshape(-1, t_candidate.shape[0], 3)
        distance = (up_c * torch.stack(v_up, dim=0)[:, None, :]).sum(dim=-1)
        id_best = distance.argmax(dim=1)
        self.v_up = t_candidate[id_best]

        self.v_up = nn.Parameter(self.v_up, requires_grad=train_up_vectors)
        return

    def register_up_vectors2(self, train_up_vectors):
        num_total_edge = np.sum([len(face) for face in self.graph1.graph["faces"]])

        id_edge_to_id_face = torch.zeros((num_total_edge, 2), dtype=torch.long)
        self.id_edge_to_id_up_dict = {}  # (id_start_point, id_end_point) -> id_up
        v_up = []

        # These three variables are used to initialize the up vector
        all_edge_points = self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer
        start_points = []
        end_points = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for idx in range(len(face_ids)):
                id_start = face_ids[idx]
                id_end = face_ids[(idx + 1) % len(face_ids)]
                id_prev = face_ids[(idx + 2) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
                    if len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 1:
                        v_up.append(-up_c)
                    else:
                        assert len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 2
                        for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                            if id_patch == key:
                                continue
                            v_up.append(torch.tensor(
                                self.graph1.edges[(id_end, id_start)]["up_c"][key], dtype=torch.float32))
                    start_points.append(all_edge_points[id_end])
                    end_points.append(all_edge_points[id_start])

        id_point_to_id_up = []
        for id_start_node in self.graph1.nodes():
            id_point_to_id_up.append([])
            for id_end_node in self.graph1[id_start_node]:
                id_point_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_start_node, id_end_node)])

        id_total_seg = 0
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for idx in range(len(face_ids)):
                id_start = face_ids[idx]
                id_end = face_ids[(idx + 1) % len(face_ids)]
                id_prev = face_ids[(idx + 2) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                id_edge_to_id_up.append([])
                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_start, id_end)])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    if len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 1:
                        v_up.append(-up_c)
                    else:
                        assert len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 2
                        for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                            if id_patch == key:
                                continue
                            v_up.append(torch.tensor(
                                self.graph1.edges[(id_end, id_start)]["up_c"][key], dtype=torch.float32))
                    start_points.append(all_edge_points[id_end])
                    end_points.append(all_edge_points[id_start])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_end, id_start)])

                id_edge_to_id_face[id_total_seg, 0] = id_patch
                key2 = -1
                for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                    if id_patch == key:
                        continue
                    key2 = key
                id_edge_to_id_face[id_total_seg, 1] = key2
                id_total_seg += 1

        self.register_buffer("id_edge_to_id_up", torch.tensor(id_edge_to_id_up))
        self.register_buffer("id_edge_to_id_face", id_edge_to_id_face)

        v_up = torch.stack(v_up, dim=0)
        start_points = torch.stack(start_points, dim=0)
        end_points = torch.stack(end_points, dim=0)

        v_normal = torch.cross(end_points - start_points, v_up)
        self.v_up = nn.Parameter(v_normal, requires_grad=train_up_vectors)
        return

    def sample_points_based_on_up(self, start_point, end_point, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))
        # cur_up = normalize_tensor(edge_up_c)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.2).to(device)  # m
        half_window_size_step = 0.01

        # Compute interpolated point
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
            edge_up_c.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        return total_num_coords, interpolated_coordinates_camera

    def sample_triangles(self, num_per_m, p1, p2, p3, num_max_sample=500):
        d1 = p2 - p1
        d2 = p3 - p2
        area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

        num_edge_points, edge_points = self.sample_edge(num_per_m,
                                                        torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)

        num_per_m2 = num_per_m * num_per_m
        num_tri_samples = torch.clamp((area * num_per_m2).to(torch.long), 1, num_max_sample * 4)
        samples = torch.rand(num_tri_samples.sum(), 2, device=p1.device)
        _t1 = torch.sqrt(samples[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * p1.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * (1 - samples[:, 1:2]) * p2.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * samples[:, 1:2] * p3.repeat_interleave(num_tri_samples, dim=0)

        # Only use the code below for debug
        if True:
            num_total_points = num_edge_points + num_tri_samples
            num_total_points_cumsum = num_total_points.cumsum(0).roll(1)
            num_total_points_cumsum[0] = 0
            sampled_total_points = torch.zeros((num_total_points.sum(), 3), device=p1.device, dtype=torch.float32)
            num_edge_points_ = num_edge_points.cumsum(0).roll(1)
            num_edge_points_[0] = 0
            num_tri_points_ = num_tri_samples.cumsum(0).roll(1)
            num_tri_points_[0] = 0
            edge_index = torch.arange(num_edge_points.sum(), device=p1.device) \
                         - (num_edge_points_ - num_total_points_cumsum).repeat_interleave(num_edge_points)
            tri_index = torch.arange(num_tri_samples.sum(), device=p1.device) \
                        - (num_tri_points_ - num_total_points_cumsum - num_edge_points).repeat_interleave(
                num_tri_samples)
            sampled_total_points[edge_index] = edge_points
            sampled_total_points[tri_index] = sampled_polygon_points
            return num_total_points, sampled_total_points
        return None, torch.cat((edge_points, sampled_polygon_points), dim=0)

    def sample_points_based_on_up_and_centroid(self, start_point, end_point, edge_up_c, v_centroid_ray):
        device = self.ray_c.device

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        plane_normal = torch.cross(cur_dir, edge_up_c)
        plane_normal = normalize_tensor(plane_normal)

        intersection_t = (start_point * plane_normal).sum(dim=1) / (v_centroid_ray * plane_normal).sum(dim=1)
        intersection_point = intersection_t[:, None] * v_centroid_ray

        num_samples, samples = self.sample_triangles(100, start_point, end_point, intersection_point)

        return num_samples, samples

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)
        return num_edge_points, sampled_edge_points

    def get_up_vector1(self, v_id_edges, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t1 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 0]]
        t2 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 1]]

        t = torch.stack([t1, t2], dim=0).reshape(-1)
        t = t * 2 * math.pi
        cur_dir = torch.stack((end_points - start_points, start_points - end_points), dim=0).reshape(-1, 3)
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            torch.sqrt(a * a + b * b) * torch.sin(t)
        ), dim=1)
        return (up_c / torch.linalg.norm(up_c.detach(), dim=-1, keepdim=True)).reshape(2, -1, 3).permute(1, 0, 2)

    def get_up_vector2(self, v_id_edges, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t1 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 0]]
        t2 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 1]]

        v_up1 = torch.cross(end_points - start_points, t1)
        v_up2 = torch.cross(start_points - end_points, t2)
        v_up = torch.stack([v_up1, v_up2], dim=0)

        return (v_up / torch.linalg.norm(v_up.detach(), dim=-1, keepdim=True)).reshape(2, -1, 3).permute(1, 0, 2)

    def compute_similarity(self, coords_c,
                           v_intrinsic1, v_transformation):
        coods2d_1 = (v_intrinsic1 @ coords_c.T).T
        coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        valid_mask1 = torch.logical_and(coods2d_1 > 0, coods2d_1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, coods2d_1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, coods2d_1[None, :, :])[0]

        # 5. Second img
        coods2d_2 = (v_transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, coods2d_2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, coods2d_2[None, :, :])[0]

        # 7. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2, reduction="none")
        return similarity_loss, torch.logical_and(valid_mask1, valid_mask2)

    def compute_similarity_wrapper(self, edge_points, up_c, v_centroid1, v_centroid2, v_mask):
        num_coors_per_edge, coords_per_edge = self.sample_points_based_on_up_and_centroid(edge_points[:, 0],
                                                                                          edge_points[:, 1],
                                                                                          up_c[:, 0], v_centroid1)
        similarity_loss11, similarity_mask1 = self.compute_similarity(coords_per_edge, self.intrinsic1,
                                                                      self.transformation)
        similarity_loss11 = scatter_add(similarity_loss11, torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss11.device).repeat_interleave(num_coors_per_edge), dim=0)
        similarity_loss11 = (similarity_loss11 / num_coors_per_edge[:, None]).mean(dim=1)
        # similarity_loss11 = similarity_loss11 * v_mask.to(torch.long)
        similarity_mask1 = scatter_min(similarity_mask1.to(torch.long), torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss11.device).repeat_interleave(num_coors_per_edge),
                                       dim=0)[0]

        num_coors_per_edge2, coords_per_edge2 = self.sample_points_based_on_up_and_centroid(edge_points[:, 1],
                                                                                            edge_points[:, 0],
                                                                                            up_c[:, 1], v_centroid2)
        similarity_loss12, similarity_mask2 = self.compute_similarity(coords_per_edge2, self.intrinsic1,
                                                                      self.transformation)
        similarity_loss12 = scatter_add(similarity_loss12, torch.arange(
            num_coors_per_edge2.shape[0], device=similarity_loss12.device).repeat_interleave(num_coors_per_edge2),
                                        dim=0)
        similarity_loss12 = (similarity_loss12 / num_coors_per_edge2[:, None]).mean(dim=1)
        similarity_loss12 = similarity_loss12 * v_mask.to(torch.long)
        similarity_mask2 = scatter_min(similarity_mask2.to(torch.long), torch.arange(
            num_coors_per_edge2.shape[0], device=similarity_loss12.device).repeat_interleave(num_coors_per_edge2),
                                       dim=0)[0]

        return similarity_loss11 + similarity_loss12, torch.cat((coords_per_edge, coords_per_edge2), dim=0), \
            torch.logical_and(similarity_mask1.to(torch.bool), similarity_mask2.to(torch.bool))

    def random_search(self, v_id_edges, v_id_points, v_centroid1, v_centroid2, v_mask):
        with torch.no_grad():
            seg_distance_detach = self.seg_distance.detach()
            ray_c = self.ray_c[v_id_points].detach()
            selected_seg_distance = seg_distance_detach[v_id_points]

            seg_distance = selected_seg_distance * self.seg_distance_normalizer
            edge_points = ray_c * seg_distance[:, :, None]
            up_c = self.get_up_vector2(v_id_edges, edge_points[:, 0], edge_points[:, 1]).detach()

            similarity_loss1, _, similarity_mask1 = self.compute_similarity_wrapper(
                edge_points, up_c, v_centroid1, v_centroid2, v_mask)
            similarity_loss1[~similarity_mask1] = torch.inf

            scale_factor = 1
            selected_seg_distance[:, 0] = -1
            while not torch.all(torch.logical_and(selected_seg_distance[:, 0] > 0, selected_seg_distance[:, 0] < 1)):
                a = seg_distance_detach[v_id_points][:, 0] + scale_factor * self.scale[v_id_points[:, 0]] * \
                    torch.distributions.utils._standard_normal(selected_seg_distance[:, 0].shape,
                                                               device=selected_seg_distance.device,
                                                               dtype=selected_seg_distance.dtype)
                mask = torch.logical_and(a > 0, a < 1)
                if mask.sum() > 0:
                    selected_seg_distance[:, 0][mask] = a[mask]

            seg_distance = selected_seg_distance * self.seg_distance_normalizer
            edge_points = ray_c * seg_distance[:, :, None]
            up_c = self.get_up_vector2(v_id_edges, edge_points[:, 0], edge_points[:, 1]).detach()

            similarity_loss2, _, similarity_mask2 = self.compute_similarity_wrapper(
                edge_points, up_c, v_centroid1, v_centroid2, v_mask)
            similarity_loss2[~similarity_mask2] = torch.inf

            sucecced_search = similarity_loss2 < similarity_loss1
            self.seg_distance.data[v_id_points[sucecced_search]] = selected_seg_distance[sucecced_search]
        return

    def forward(self, v_ids, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1

        v_id_edge = v_ids[0]  # (num_edge, )
        v_id_edge_points = v_ids[1]  # (num_edge, 4)

        v_id_faces = self.id_edge_to_id_face[v_id_edge]

        centroid_ray1 = self.center_ray_c[v_id_faces[:, 0]]
        centroid_ray2 = self.center_ray_c[v_id_faces[:, 1]]
        mask = v_id_faces[:, 1] != -1
        # Random search
        if self.training:
            self.random_search(v_id_edge, v_id_edge_points, centroid_ray1, centroid_ray2, mask)

        ray_c = self.ray_c[v_id_edge_points]
        seg_distance = self.seg_distance[v_id_edge_points] * self.seg_distance_normalizer
        edge_points = ray_c * seg_distance[:, :, None]
        up_c = self.get_up_vector2(v_id_edge, edge_points[:, 0].detach(), edge_points[:, 1].detach())
        # edge_points[:,1:] = edge_points[:,1:].detach()

        similarity_loss, coords_per_edge, similarity_mask = self.compute_similarity_wrapper(
            edge_points, up_c, centroid_ray1, centroid_ray2, mask)
        similarity_loss = similarity_loss.mean()
        if is_log and self.id_viz_edge in v_id_edge:
            with torch.no_grad():
                line_thickness = 1
                point_thickness = 2
                point_radius = 1

                polygon_points_2d_1 = (self.intrinsic1 @ coords_per_edge.T).T
                polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
                polygon_points_2d_2 = (self.transformation @ to_homogeneous_tensor(coords_per_edge).T).T
                polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()

                line_img1 = self.rgb1.copy()
                line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
                shape = line_img1.shape[:2][::-1]

                roi_coor_2d1_numpy = np.clip(polygon_points_2d_1, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                polygon_2d1 = (self.intrinsic1 @ edge_points[0].T).T
                polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
                polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.line(line_img1, polygon_2d1[0], polygon_2d1[1],
                         color=(0, 255, 0), thickness=line_thickness)
                cv2.circle(line_img1, polygon_2d1[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img1, polygon_2d1[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

                # Image 2
                line_img2 = self.rgb2.copy()
                line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                shape = line_img2.shape[:2][::-1]

                roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                polygon_2d2 = (self.transformation @ to_homogeneous_tensor(edge_points[0]).T).T
                polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
                polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.line(line_img2, polygon_2d2[0], polygon_2d2[1],
                         color=(0, 255, 0), thickness=line_thickness)
                cv2.circle(line_img2, polygon_2d2[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img2, polygon_2d2[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))

        weighted_similarity = similarity_loss * 1
        # weighted_edge = edge_loss * (-1)
        total_loss = weighted_similarity

        return total_loss, [None, None, None]

    def debug_save(self, v_index):
        id_epoch = v_index + 1
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_points, v_up_c):
            total_edge_points = v_edge_points

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + v_up_c

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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        id_patch = torch.tensor((self.id_viz_face,), dtype=torch.long, device=point_pos_c.device)
        # Visualize target patch
        edge_points = point_pos_c[self.batched_points_per_patch[id_patch]].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(np.arange(self.id_viz_edge, self.id_viz_edge + edge_points.shape[0]),
                                   edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(id_epoch)), arrows)
        id_points = np.asarray(self.batched_points_per_patch[id_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(id_epoch)), start_point_w, edge_index)
        return 0

        # Visualize whole patch
        edge_points = point_pos_c[list(itertools.chain(*self.batched_points_per_patch))].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(np.arange(np.sum([len(item) // 4 for item in self.batched_points_per_patch])),
                                   edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)), start_point_w, edge_index)
        pass

        return 0

    def len(self):
        return len(self.graph1.graph["faces"])



def get_up_vector(start_points, end_points, t):
    # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
    cur_dir = end_points - start_points
    a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
    up_c = torch.stack((
        -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
        a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
        torch.sqrt(a * a + b * b) * torch.sin(t)
    ), dim=1)
    return normalize_tensor(up_c)


def sample_new_ups(v_edge_points):
    num_edges = v_edge_points.shape[0]
    t = torch.linspace(0, 1, steps=100, device=v_edge_points.device) * torch.pi * 2
    num_sample = t.shape[0]

    sampled_ups = get_up_vector(
        v_edge_points[:, None, 0, :].tile(1, num_sample, 1).view(-1, 3),
        v_edge_points[:, None, 1, :].tile(1, num_sample, 1).view(-1, 3),
        t[None, :].tile(num_edges, 1).view(-1),
    ).view(num_edges, num_sample, 3)

    return sampled_ups


def compute_similarity_wrapper(start_points, end_points,
                               imgs, transformations, intrinsic, v_vertical_length=10):
    times = [0 for _ in range(10)]
    cur_time = time.time()
    num_src_imgs = imgs.shape[0] - 1
    points_c = torch.stack([start_points, end_points], dim=1).reshape(-1, 3)
    times[0] += time.time() - cur_time
    cur_time = time.time()

    edge_points = (intrinsic @ points_c.T).T
    edge_points = edge_points[:, :2] / (edge_points[:, 2:3] + 1e-6)
    edge_points = edge_points.reshape(-1, 2, 2)

    # sample step=0.01
    num_horizontal = torch.clamp((torch.linalg.norm(end_points - start_points, dim=-1) / 0.01).to(torch.long),
                                 2, 1000)

    num_per_edge1, points_2d1 = sample_points_2d(edge_points, num_horizontal, imgs[0].shape[1],
                                                 v_vertical_length=v_vertical_length)

    valid_mask1 = torch.logical_and(points_2d1 > 0, points_2d1 < 1)
    valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
    points_2d1 = torch.clamp(points_2d1, 0, 0.999999)

    edge_points = (transformations @ to_homogeneous_tensor(points_c).T).transpose(1, 2)
    edge_points = edge_points[:, :, :2] / (edge_points[:, :, 2:3] + 1e-6)
    edge_points = edge_points.reshape(num_src_imgs, -1, 2, 2)

    num_per_edge2, points_2d2 = sample_points_2d(edge_points.reshape(-1, 2, 2),
                                                 num_horizontal.tile(num_src_imgs), imgs[0].shape[1],
                                                 v_vertical_length=v_vertical_length)
    num_per_edge2 = num_per_edge2.reshape(num_src_imgs, -1)
    points_2d2 = points_2d2.reshape(num_src_imgs, -1, 2)

    valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
    valid_mask2 = torch.logical_and(valid_mask2[:, :, 0], valid_mask2[:, :, 1])
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    # 4. Sample pixel color
    sample_imgs1 = sample_img(imgs[0:1, None, :, :], points_2d1[None, :, :])[0]
    sample_imgs2 = sample_img(imgs[1:, None], points_2d2)
    times[1] += time.time() - cur_time
    cur_time = time.time()

    similarity_loss, black_area_in_img1 = loss4(sample_imgs1, sample_imgs2, num_per_edge1)
    similarity_mask = torch.logical_and(valid_mask1[None, :].tile([valid_mask2.shape[0], 1]), valid_mask2)
    similarity_mask = scatter_min(similarity_mask.to(torch.long), torch.arange(
        num_per_edge1.shape[0], device=similarity_loss.device).repeat_interleave(num_per_edge1), dim=1)[0]
    times[3] += time.time() - cur_time
    cur_time = time.time()
    return similarity_loss, similarity_mask.to(torch.bool), black_area_in_img1, [points_2d1, points_2d2]


def get_initial_normal(v_center_ray):
    return v_center_ray


def sample_new_directions_batch(v_input, num_sample, v_variance):
    batch_size = v_input.shape[0]
    device = v_input.device
    v_input = v_input / (v_input.norm(dim=1, keepdim=True) + 1e-6)

    # Generate N random unit vectors for each input vector
    random_vectors = torch.randn(batch_size, num_sample, 3, device=device)
    random_vectors = random_vectors / (torch.norm(random_vectors, dim=2, keepdim=True) + 1e-6)

    # Project random vectors onto the plane orthogonal to the input vectors
    projections = torch.matmul(random_vectors, v_input.unsqueeze(2))
    orthogonal_vectors = random_vectors - projections * v_input.unsqueeze(1)

    # Generate Gaussian distributed angles
    angles = torch.randn(batch_size, num_sample, device=device) * math.sqrt(v_variance)

    # Calculate the new directions
    new_directions = torch.cos(angles).unsqueeze(2) * v_input.unsqueeze(1) + torch.sin(angles).unsqueeze(
        2) * orthogonal_vectors

    new_directions = normalize_tensor(new_directions)
    return new_directions


def sample_new_distance_and_normals(v_center_distance, v_init_normal, num_sample=100, v_max=10., scale_factor=1.6, ):
    num_vertices = v_center_distance.shape[0]
    device = v_center_distance.device
    dtype = v_center_distance.dtype
    # (B, S)
    new_distance = -torch.ones((num_vertices, num_sample - 1), device=device, dtype=dtype)
    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, S)
    repeated_vertices_distances = v_center_distance[:, None].tile((1, num_sample - 1))
    while not torch.all(sample_distance_mask):
        t_ = new_distance[~sample_distance_mask]
        a = repeated_vertices_distances[~sample_distance_mask] + \
            scale_factor * torch.distributions.utils._standard_normal(
            t_.shape[0],
            device=device,
            dtype=t_.dtype)
        new_distance[~sample_distance_mask] = a
        sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, (S + 1))
    new_distance = torch.cat((v_center_distance[:, None], new_distance), dim=1)

    new_normals = -torch.ones((num_vertices, num_sample - 1, 3), device=device, dtype=dtype)
    sample_normals_mask = torch.zeros_like(new_normals[:, :, 0], dtype=torch.bool)
    while not torch.all(sample_normals_mask):
        a = sample_new_directions_batch(v_init_normal, num_sample - 1, 0.3)
        new_normals[~sample_normals_mask] = a[~sample_normals_mask]
        sample_normals_mask = (v_init_normal[:, None, :] * new_normals).sum(dim=-1) > 0

    new_normals = sample_new_directions_batch(v_init_normal, num_sample - 1, 0.3)
    new_normals = torch.cat((v_init_normal[:, None, :], new_normals), dim=1)
    return new_distance, new_normals


def get_edge_vertices(v_center_point, v_normal, src_rays):
    # Calculate the distance from the origin to the plane along the normal vector
    d = torch.sum(v_center_point * v_normal, dim=2)
    # Calculate the dot product between the rays (a) and the normal vectors
    a_dot_v_normal = torch.matmul(src_rays, v_normal.transpose(1, 2))
    # Calculate the t values for each ray-plane intersection
    t = d[:, None, :] / a_dot_v_normal
    collision_points = src_rays[:, :, None, :] * t[:, :, :, None]
    return collision_points


def evaluate_candidates(v_edge_points, v_intrinsic, v_transformation,
                        v_img1, v_img2, v_sample_density=0.01):
    num_edge, _, _ = v_edge_points.shape

    num_horizontal = torch.clamp(
        (torch.linalg.norm(v_edge_points[:, 0] - v_edge_points[:, 1], dim=-1) / v_sample_density).to(torch.long),
        2, 1000)

    points_2d1 = (v_intrinsic @ v_edge_points.view(-1, 3).T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-6)
    points_2d1 = points_2d1.reshape(num_edge, 2, 2)

    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_edge_points.view(-1, 3)).T).transpose(0, 1)
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-6)
    points_2d2 = points_2d2.reshape(num_edge, 2, 2)
    valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])  # And in two edge points
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])  # And in x and y
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    num_sampled_points1, sampled_points1 = sample_points_2d(points_2d1.reshape(-1, 2, 2),
                                                            num_horizontal.reshape(-1),
                                                            v_vertical_length=5,
                                                            v_img_width=v_img1.shape[1])
    _, sampled_points2 = sample_points_2d(points_2d2.reshape(-1, 2, 2),
                                          num_horizontal.reshape(-1),
                                          v_img_width=v_img2.shape[1])

    # 4. Sample pixel color
    sample_imgs1 = sample_img(v_img1[None, None, :], sampled_points1[None, :, :])[0]
    sample_imgs2 = sample_img(v_img2[None, None, :], sampled_points2[None, :, :])[0]

    similarity_loss = loss4(sample_imgs1, sample_imgs2[None, :], num_sampled_points1)[0]
    similarity_loss = similarity_loss[0]
    similarity_loss = similarity_loss.reshape(num_edge, )

    # Visualize
    is_viz = False
    if is_viz:
        num_samples = num_sampled_points1.reshape(36, 100, 3)
        for id_point in range(num_samples.shape[0]):
            img1 = cv2.cvtColor((v_img1.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor((v_img2.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            id_start = num_sampled_points1[:100 * 3 * id_point].sum()
            coords1 = sampled_points1[id_start:id_start + num_samples[id_point, 0].sum()]
            coords1 = (coords1 * 800).cpu().numpy().astype(np.int64)
            coords1 = np.clip(coords1, 0, 799)
            img1[coords1[:, 1], coords1[:, 0]] = (0, 0, 255)
            coords2 = sampled_points2[id_start:id_start + num_samples[id_point, 0].sum()]
            coords2 = (coords2 * 800).cpu().numpy().astype(np.int64)
            coords2 = np.clip(coords2, 0, 799)
            img2[coords2[:, 1], coords2[:, 0]] = (0, 0, 255)

            similarity_loss_ = similarity_loss.reshape(36, 100, 3).cpu().numpy()

            print("{:.2f}, {:.2f}, {:.2f}; {}, {}, {}".format(
                similarity_loss_[id_point, 0, 0], similarity_loss_[id_point, 0, 1], similarity_loss_[id_point, 0, 2],
                num_samples[id_point, 0, 0], num_samples[id_point, 0, 1], num_samples[id_point, 0, 2]
            ))

            for i in range(3):
                img11 = sample_imgs1[id_start:id_start + num_samples[id_point, 0, i]].reshape(-1, 20).cpu().numpy()
                img11 = (img11 * 255).astype(np.uint8)
                img22 = sample_imgs2[id_start:id_start + num_samples[id_point, 0, i]].reshape(-1, 20).cpu().numpy()
                img22 = (img22 * 255).astype(np.uint8)
                cv2.imwrite("output/{}.jpg".format(i), np.concatenate((img11, img22), axis=1))
                id_start = id_start + num_samples[id_point, 0, i]

            cv2.imshow("1", np.concatenate((img1, img2), axis=1))
            cv2.waitKey(0)

        pass

    return similarity_loss, valid_mask2, num_sampled_points1


def optimize1(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vertical_length = 5

    for id_img1, graph in enumerate(v_graphs):
        all_face_ids = np.asarray(graph.graph["faces"], dtype=object)

        rays_c = [None] * len(graph.nodes)
        distances = [None] * len(graph.nodes)
        id_end = []
        for idx, id_node in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_node]["ray_c"]
            distances[idx] = graph.nodes[id_node]["distance"]
            id_end.append(list(graph[0].keys()))

        id_src_imgs = v_img_pairs[id_img1][:, 0]
        ref_img = cv2.imread(v_img_database[id_img1].img_path,
                             cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path,
                               cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Vertex
        num_vertex = len(rays_c)
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        distances = torch.from_numpy(np.stack(distances)).to(device).to(torch.float32)
        # Face
        face_flags = torch.tensor(graph.graph["face_flags"], dtype=torch.bool, device=device)
        centers_ray_c = torch.from_numpy(graph.graph["ray_c"]).to(device).to(torch.float32)
        center_distances = torch.from_numpy(graph.graph["distance"]).to(device).to(torch.float32)

        point_id_per_edge = []
        edge_id_dict = {}
        for edge in graph.edges():
            point_id_per_edge.append(edge)
            edge_id_dict[(edge[0], edge[1])] = len(point_id_per_edge)
            edge_id_dict[(edge[1], edge[0])] = -len(point_id_per_edge)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        # Edges
        num_edge = len(graph.edges())
        num_max_edge_per_vertex = max([len(graph[item]) for item in graph.nodes])
        edge_rays_c = rays_c[point_id_per_edge]
        edge_distances_c = distances[point_id_per_edge]
        edge_is_not_black = determine_valid_edges(edge_rays_c, edge_distances_c, imgs[0], intrinsic)[:, 0]
        to_non_black_edge_id = -torch.ones(num_edge, device=device, dtype=torch.long)
        to_non_black_edge_id[edge_is_not_black] = torch.arange(edge_is_not_black.sum(), device=device)

        edge_point_id_per_face = []
        for face_ids, face_flag in zip(all_face_ids, face_flags):
            if not face_flag:
                continue
            local_id_edges = []
            for i, id_start in enumerate(face_ids):
                id_end = face_ids[(i + 1) % len(face_ids)]
                id_edge = edge_id_dict[(id_start, id_end)]
                mask = 1 if id_edge > 0 else -1
                local_id_edges.append((to_non_black_edge_id[abs(id_edge) - 1] + 1) * mask)
            a = torch.tensor(local_id_edges, dtype=torch.long, device=device)
            b = torch.roll(a, -1)
            edge_point_id_per_face.append(torch.stack((a, b), dim=1))

        edge_rays_c = edge_rays_c[edge_is_not_black]
        edge_distances_c = edge_distances_c[edge_is_not_black]
        num_edge = edge_distances_c.shape[0]
        num_sample = 100

        # edge_rays_c = edge_rays_c[1:2]
        # edge_distances_c = edge_distances_c[1:2]
        # num_edge = 1

        cur_iter = 0
        while True:
            distances_candidate = sample_new_distance(edge_distances_c.view(-1), num_sample=num_sample).view(
                num_edge, 2, num_sample, )
            # num_edge, 2, num_sample, 3
            edge_points_candidate = edge_rays_c[:, :, None, :] * distances_candidate[:, :, :, None]

            edge_points_candidate = edge_rays_c[:, :, None, :] * distances_candidate[:, :, :, None]
            ncc_loss, ncc_loss_mask = evaluate_candidates(edge_points_candidate, intrinsic,
                                                          transformation[0], imgs[0], imgs[1])
            ncc_loss[~ncc_loss_mask] = torch.inf  # Set it to 0 in order to get the samples below

            # Consistent loss
            sampled_index = torch.multinomial(torch.exp(-ncc_loss ** 4), 99, replacement=True)
            sampled_index = torch.cat((torch.zeros_like(sampled_index[:, 0:1]), sampled_index), dim=1)
            sampled_loss = torch.gather(ncc_loss, dim=1, index=sampled_index).mean(dim=0)
            sampled_edges = torch.gather(edge_points_candidate, dim=2,
                                         index=sampled_index.view(num_edge, 1, 100, 1).tile(1, 2, 1, 3))
            sampled_edges = sampled_edges.permute(2, 0, 1, 3)
            consistency = []
            for face in edge_point_id_per_face:
                true_face = face[torch.all(face != 0, dim=1)]
                if true_face.shape[0] == 0:
                    continue
                face_mask = true_face < 0
                chosen_points = sampled_edges[:, torch.abs(true_face) - 1]
                chosen_points[:, face_mask] = torch.flip(chosen_points[:, face_mask], dims=[2])
                distance = torch.linalg.norm(chosen_points[:, :, 0, 1] - chosen_points[:, :, 1, 0], dim=2).mean(dim=1)
                consistency.append(distance)
            consistency = torch.stack(consistency, dim=1).mean(dim=1)

            alpha = 1
            loss = sampled_loss * alpha + consistency * (1 - alpha)
            id_best = sampled_index[:, loss.argmin(dim=0)]
            chosen_distance = torch.gather(distances_candidate, dim=2, index=id_best.view(-1, 1, 1).tile(1, 2, 1))[:, :,
                              0]
            chosen_edge_points = torch.gather(edge_points_candidate, dim=2,
                                              index=id_best.view(-1, 1, 1, 1).tile(1, 2, 1, 3))[:, :, 0]
            chosen_loss = torch.gather(loss, dim=0, index=loss.argmin(dim=0))
            # chosen_edge_points = edge_points_candidate[:,:,0]
            edge_distances_c = chosen_distance
            # Log
            if True:
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                start_points_c = chosen_edge_points[:, 0]
                end_points_c = chosen_edge_points[:, 1]
                sp_2d1 = ((intrinsic @ start_points_c.T).T).cpu().numpy()
                ep_2d1 = ((intrinsic @ end_points_c.T).T).cpu().numpy()
                sp_2d2 = (transformation[0] @ to_homogeneous_tensor(start_points_c).T).T.cpu().numpy()
                ep_2d2 = (transformation[0] @ to_homogeneous_tensor(end_points_c).T).T.cpu().numpy()
                sp_2d1 = sp_2d1[:, :2] / sp_2d1[:, 2:3]
                ep_2d1 = ep_2d1[:, :2] / ep_2d1[:, 2:3]
                sp_2d2 = sp_2d2[:, :2] / sp_2d2[:, 2:3]
                ep_2d2 = ep_2d2[:, :2] / ep_2d2[:, 2:3]

                shape = img11.shape[:2][::-1]
                sp_2d1 = np.around(sp_2d1 * shape).astype(np.int64)
                ep_2d1 = np.around(ep_2d1 * shape).astype(np.int64)
                sp_2d2 = np.around(sp_2d2 * shape).astype(np.int64)
                ep_2d2 = np.around(ep_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(sp_2d1.shape[0]):
                    cv2.line(img11, sp_2d1[i_edge], ep_2d1[i_edge], line_color, line_thickness)
                    cv2.line(img21, sp_2d2[i_edge], ep_2d2[i_edge], line_color, line_thickness)
                for i_point in range(sp_2d1.shape[0]):
                    cv2.circle(img11, sp_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img11, ep_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, sp_2d2[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, ep_2d2[i_point], 1, point_color, point_thickness)

                cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                            np.concatenate([
                                np.concatenate([img11, img21], axis=1),
                                np.concatenate([img12, img22], axis=1),
                            ], axis=0)
                            )
                print("{:3d}:{:.4f}".format(cur_iter,
                                            chosen_loss.mean().cpu().item()))

            cur_iter += 1
        exit()

    pass


def sample_img_window(v_rgb, v_coor_2d):
    MAX_ITEM_PER_BATCH = 262144
    window_size = 7  # 窗口大小
    padding = (window_size - 1) // 2  # 计算填充大小
    v_rgb_padded = torch.nn.functional.pad(v_rgb, (padding, padding, padding, padding), mode='constant',
                                           value=0)  # 对图像进行填充

    # 计算归一化后的窗口大小，用于torch.grid_sample的输入
    img_height, img_width = v_rgb.shape[-2:]
    window_size_normalized = window_size / (img_width - 1), window_size / (img_height - 1)

    # 创建网格以获得7x7窗口
    grid_x, grid_y = torch.meshgrid(torch.linspace(-window_size_normalized[0], window_size_normalized[0], window_size),
                                    torch.linspace(-window_size_normalized[1], window_size_normalized[1], window_size))
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # 将x和y网格合并为一个网格张量
    grid = grid.to(v_coor_2d.device)

    coor_2d = v_coor_2d
    results = []
    for i in range(ceil(coor_2d.shape[1] / MAX_ITEM_PER_BATCH)):
        batch_coor_2d = coor_2d[:,
                        i * MAX_ITEM_PER_BATCH:min(coor_2d.shape[1], (i + 1) * MAX_ITEM_PER_BATCH)]  # 获取当前批次的坐标
        batch_coor_2d = batch_coor_2d.unsqueeze(2).unsqueeze(3).tile(1, 1, window_size, window_size,
                                                                     1)  # (n, num_points, window_size, window_size, 2)
        # TODO: fix the sample points in the edge of img(now it's clampped)
        batch_grid = (grid + batch_coor_2d * 2 - 1).clamp(-1, 1)  # 将网格加到坐标上，并将范围限制在[-1, 1]
        batch_grid = batch_grid.view(batch_grid.shape[0], batch_grid.shape[1], -1,
                                     2)  # (n, num_points, window_size*window_size, 2)
        # 使用grid_sample从输入图像中采样7x7窗口
        sampled_pixels = torch.nn.functional.grid_sample(v_rgb_padded, batch_grid,
                                                         align_corners=True)  # 输出维度为(1, 1, k, 49)
        results.append(sampled_pixels)

    sampled_pixels = torch.cat(results, dim=2).reshape(1, -1, window_size, window_size)  # 调整输出张量维度为(1, k, 7, 7)
    return sampled_pixels


def optimize(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vertical_length = 5

    for id_img1, graph in enumerate(v_graphs):
        # The vertex id for each face
        all_face_ids = np.asarray(graph.graph["faces"], dtype=object)

        # The ray of each node in camera coordinate
        rays_c = [None] * len(graph.nodes)
        # The distance of each node along the ray direction in camera coordinate
        ray_distances_c = [None] * len(graph.nodes)
        # id_end = []
        for idx, id_points in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_points]["ray_c"]
            ray_distances_c[idx] = graph.nodes[id_points]["distance"]
            # id_end.append(list(graph[0].keys()))

        # prepare some data
        id_src_imgs = v_img_pairs[id_img1][:, 0]
        ref_img = cv2.imread(v_img_database[id_img1].img_path, cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path, cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.
        # transformation store the transformation matrix from ref_img to src_imgs
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        c1_2_c2 = torch.from_numpy(
            v_img_database[int(id_src_imgs[0])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        ).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Vertex
        num_vertex = len(rays_c)
        # rays_c = query_points(nodes)' pos in camera coordinate
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        ray_distances_c = torch.from_numpy(np.stack(ray_distances_c)).to(device).to(torch.float32)
        # Face
        # face_flag record if it's a valid face
        face_flags = torch.tensor(graph.graph["face_flags"], dtype=torch.bool, device=device)
        centroid_rays_c = torch.from_numpy(graph.graph["ray_c"]).to(device).to(torch.float32)
        centroid_ray_distances_c = torch.from_numpy(graph.graph["distance"]).to(device).to(torch.float32)

        def prepare_edge_data(v_graph):
            determine_valid_edges(v_graph, imgs[0])
            # The id of the vertex for all edges
            point_id_per_edge = []
            # The id of a specific edge in the `point_id_per_edge`
            edge_id_dict = {}
            for edge in v_graph.edges():
                if not v_graph.edges[edge]["valid_flag"] or v_graph.edges[edge]["is_black"]:
                    continue
                point_id_per_edge.append(edge)
                # + and - to distinguish whether the edge is flip
                edge_id_dict[(edge[0], edge[1])] = len(point_id_per_edge)
                edge_id_dict[(edge[1], edge[0])] = -len(point_id_per_edge)
            return point_id_per_edge, edge_id_dict

        point_id_per_edge, edge_id_dict = prepare_edge_data(graph)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        def prepare_node_data(v_point_id_per_edge, v_device):
            unique_point_ids, counts = torch.unique(v_point_id_per_edge, return_counts=True)
            unique_point_ids = unique_point_ids[counts > 1]
            num_point = unique_point_ids.shape[0]
            max_counts = counts.max()

            node_data = -torch.ones((num_point, max_counts + 1), device=v_device, dtype=torch.long)
            for idx, id_point in enumerate(unique_point_ids):
                coords = torch.stack(torch.where(v_point_id_per_edge == id_point), dim=1)
                num_valid = coords.shape[0]
                coords[:, 1] = 1 - coords[:, 1]  # Index another point
                node_data[idx, 0] = id_point
                node_data[idx, 1:num_valid + 1] = v_point_id_per_edge[coords[:, 0], coords[:, 1]]
            return node_data

        # Index the valid nodes and the corresponding edges
        # The node_data is in the shape of [M,K],
        # where M is the number of valid node, K is the maximum edges (K=2) means every node has at most 1 edge
        # invalid edges are padded with -1
        node_data = prepare_node_data(point_id_per_edge, device)

        # Visualize the nodes that to be optimized
        if False:
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]

            # Draw edges
            for node in node_data:
                id_start = int(node[0].cpu().item())
                for id_end in node[1:]:
                    id_end = int(id_end.cpu().item())
                    if id_end == -1:
                        break
                    start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                    end_pos = np.around((graph.nodes[id_end]["pos_2d"]) * shape).astype(np.int64)
                    cv2.line(img11, start_pos, end_pos, (0, 0, 255), 2)

            # Draw nodes
            for node in node_data:
                id_start = int(node[0].cpu().item())
                start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                cv2.circle(img11, start_pos, 2, (0, 255, 255), 2)

            cv2.imshow("1", img11)
            cv2.waitKey(0)

        img_src_id = 0

        def optimize_node(v_node_data, v_rays_c, v_ray_distances_c):
            id_start_points = v_node_data[:, 0]
            id_end_points = v_node_data[:, 1:]
            # M
            num_vertex = id_start_points.shape[0]
            # N
            num_max_edge_per_vertex = id_end_points.shape[1]

            optimized_distance = v_ray_distances_c.clone()
            # (M, N)
            end_distances = optimized_distance[id_end_points]
            # K
            num_sample = 100

            # (M, K, N)
            edge_valid_flag = (id_end_points == -1)[:, None, :].tile(1, num_sample, 1)

            cur_iter = 0
            last_loss = None
            num_tolerence = 100
            while True:
                # (M, K)
                start_distances_candidate = sample_new_distance(optimized_distance[id_start_points],
                                                                num_sample=num_sample).view(
                    num_vertex, num_sample, )
                # (M, K, 3)
                start_points_c = start_distances_candidate[:, :, None] * v_rays_c[id_start_points][:, None, :]
                # (M, K, N, 3)
                start_points_c = start_points_c[:, :, None, :].tile(1, 1, num_max_edge_per_vertex, 1)
                # (M, K, N, 3)
                end_points_c = (end_distances[:, :, None] * v_rays_c[id_end_points])[:, None, :, :].tile(1, 100, 1, 1)

                # (M, K, N, 2, 3)
                edges_points = torch.stack((start_points_c, end_points_c), dim=-2)

                ncc_loss, ncc_loss_mask, num_samples_per_edge = evaluate_candidates(edges_points.reshape(-1, 2, 3),
                                                                                    intrinsic,
                                                                                    transformation[img_src_id],
                                                                                    imgs[0], imgs[img_src_id + 1],
                                                                                    v_sample_density=0.01)

                ncc_loss[~ncc_loss_mask] = torch.inf
                ncc_loss = ncc_loss.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                num_samples_per_edge = num_samples_per_edge.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                ncc_loss[edge_valid_flag] = 0
                num_samples_per_edge[edge_valid_flag] = 0
                # ncc_loss = ncc_loss * num_samples_per_edge/num_samples_per_edge.sum(dim=-1,keepdim=True)
                ncc_loss = ncc_loss.mean(dim=-1)
                id_best = ncc_loss.argmin(dim=1)

                optimized_distance[id_start_points] = torch.gather(start_distances_candidate, 1, id_best[:, None])[:, 0]
                end_distances = optimized_distance[id_end_points]

                total_loss = torch.gather(ncc_loss, 1, id_best[:, None]).mean()

                if last_loss is None:
                    last_loss = total_loss
                else:
                    delta = last_loss - total_loss

                    print("{:3d}:{:.4f}; Delta:{:.4f}".format(cur_iter,
                                                              total_loss.cpu().item(),
                                                              delta.cpu().item()
                                                              ))

                    if delta < 1e-5:
                        num_tolerence -= 1
                    else:
                        num_tolerence = 100
                        last_loss = total_loss
                    if num_tolerence <= 0:
                        break

                # Visualize
                if True:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)
                    img22 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)

                    start_points_c = optimized_distance[id_start_points][:, None] * v_rays_c[id_start_points]
                    start_points_c = start_points_c[:, None, :, ].tile(1, num_max_edge_per_vertex, 1)
                    end_points_c = optimized_distance[id_end_points][:, :, None] * v_rays_c[id_end_points]

                    all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                    all_points_c = all_points_c[~edge_valid_flag[:, 0]]
                    all_points_c = all_points_c.reshape(-1, 3)
                    p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                    p_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                    p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                    p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                    p_2d1 = p_2d1.reshape(-1, 2, 2)
                    p_2d2 = p_2d2.reshape(-1, 2, 2)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                    p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                    line_color = (0, 0, 255)
                    line_thickness = 2
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_edge in range(p_2d1.shape[0]):
                        cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                        cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                    cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                                np.concatenate([
                                    np.concatenate([img11, img21], axis=1),
                                    np.concatenate([img12, img22], axis=1),
                                ], axis=0)
                                )

                cur_iter += 1
            return optimized_distance

        optmized_distance_cache = "output/optimized_distance.tar"
        if os.path.exists(optmized_distance_cache):
            optimized_distance = torch.load(optmized_distance_cache).to(device)
        else:
            optimized_distance = optimize_node(node_data, rays_c, ray_distances_c)
            torch.save(optimized_distance, optmized_distance_cache)

        # Remove the redundant face and edges in the graph
        # And build the dual graph in order to navigate between patches
        def fix_graph(v_graph, is_visualize=False):
            dual_graph = nx.Graph()

            id_original_to_current = {}
            for id_face, face in enumerate(graph.graph["faces"]):

                if not graph.graph["face_flags"][id_face]:
                    continue
                has_black = False
                for idx, id_start in enumerate(face):
                    id_end = face[(idx + 1) % len(face)]
                    if v_graph.edges[(id_start, id_end)]["is_black"]:
                        has_black = True
                        break
                if has_black:
                    continue

                id_original_to_current[id_face] = len(dual_graph.nodes)
                dual_graph.add_node(len(dual_graph.nodes), id_vertex=face,
                                    id_in_original_array=id_face,
                                    face_center=graph.graph['face_center'][id_face],
                                    ray_c=graph.graph['ray_c'][id_face])

            for node in dual_graph.nodes():
                faces = dual_graph.nodes[node]["id_vertex"]
                for idx, id_start in enumerate(faces):
                    id_end = faces[(idx + 1) % len(faces)]
                    t = copy(graph.edges[(id_start, id_end)]["id_face"])
                    t.remove(dual_graph.nodes[node]["id_in_original_array"])
                    if t[0] in id_original_to_current:
                        edge = (node, id_original_to_current[t[0]])
                        if edge not in dual_graph.edges():
                            id_points_in_another_face = dual_graph.nodes[id_original_to_current[t[0]]]["id_vertex"]
                            adjacent_vertices = []
                            id_cur = idx
                            while True:
                                adjacent_vertices.append(id_start)
                                id_cur += 1
                                if id_cur >= len(faces):
                                    adjacent_vertices.append(id_end)
                                    break
                                id_start = faces[id_cur]
                                id_end = faces[(id_cur + 1) % len(faces)]
                                if id_end not in id_points_in_another_face:
                                    adjacent_vertices.append(id_start)
                                    break

                            # adjacent_vertices.append(id_start)
                            dual_graph.add_edge(edge[0], edge[1], adjacent_vertices=adjacent_vertices)

            v_graph.graph["dual_graph"] = dual_graph

            # Visualize
            if is_visualize:
                for idx, id_face in enumerate(dual_graph.nodes):
                    print("{}/{}".format(idx, len(dual_graph.nodes)))
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    shape = img11.shape[:2][::-1]
                    face = dual_graph.nodes[id_face]["id_vertex"]

                    for idx, id_start in enumerate(face):
                        id_end = face[(idx + 1) % len(face)]
                        pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                        pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                        cv2.line(img11, pos1, pos2, (0, 0, 255), 2)

                    for id_another_face in dual_graph[id_face]:
                        face = dual_graph.nodes[id_another_face]["id_vertex"]
                        for idx, id_start in enumerate(face):
                            id_end = face[(idx + 1) % len(face)]
                            pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                            pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                            cv2.line(img11, pos1, pos2, (0, 255, 255), 1)

                        for id_vertex in dual_graph[id_face][id_another_face]["adjacent_vertices"]:
                            pos1 = np.around(v_graph.nodes[id_vertex]["pos_2d"] * shape).astype(np.int64)
                            cv2.circle(img11, pos1, 2, (0, 255, 0), 2)

                    cv2.imshow("1", img11)
                    cv2.waitKey()
            return

        fix_graph(graph)

        # fun defined by xdt
        def fit_plane_svd(points: torch.Tensor) -> torch.Tensor:
            centroid = torch.mean(points, axis=0)
            centered_points = points - centroid
            u, s, vh = torch.linalg.svd(centered_points)
            d = -torch.dot(vh[-1], centroid)
            abcd = torch.cat((vh[-1], torch.tensor([d]).to(device)))
            return abcd

        # TODO: using ray_c
        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def ncc_matching_cost(img1, img2):
            mean1 = torch.mean(img1)
            mean2 = torch.mean(img2)
            std1 = torch.std(img1)
            std2 = torch.std(img2)
            norm_img1 = (img1 - mean1) / std1
            norm_img2 = (img2 - mean2) / std2
            ncc = torch.sum(norm_img1 * norm_img2) / (img1.numel() - 1)
            matching_cost = 1 - ncc
            return matching_cost

        def vectors_to_angles(normal_vectors):
            normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
            x, y, z = normal_vectors.unbind(dim=1)
            phi = torch.atan2(y, x)
            theta = torch.acos(z)
            return torch.stack([phi, theta], dim=1)

        def angles_to_vectors(angles):
            phi, theta = angles.unbind(dim=1)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            return torch.stack([x, y, z], dim=1)

        # 对应位置求交点
        def intersection_of_ray_and_plane(planes, rays_direction):
            # planes: n*4 ray: n*3
            n = planes[:, :3]
            d = planes[:, 3]
            denominator = torch.sum(n * rays_direction, dim=1)
            t = -d / torch.sum(n * rays_direction, dim=1)
            intersection_points = torch.unsqueeze(t, 1) * rays_direction
            valid_intersection = (denominator != 0) & (t >= 0)
            # n * _
            return valid_intersection, intersection_points

        def bilateral_ncc_(v_img1, v_img2):
            batch_size = v_img1.shape[0]
            window_size = v_img1.shape[1]
            device = v_img1.device
            # Add weights
            sigma_spatial = 3
            spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
            spatial_weights = torch.stack(torch.meshgrid(
                torch.arange(window_size, device=device, dtype=torch.float32),
                torch.arange(window_size, device=device, dtype=torch.float32),
                indexing="xy"
            ), dim=2) - window_size // 2
            spatial_weights = torch.linalg.norm(spatial_weights, dim=-1)
            spatial_weights = spatial_weights ** 2 * spatial_normalization_
            spatial_weights = torch.exp(-spatial_weights)
            spatial_weights = spatial_weights / spatial_weights.sum()

            v_img1 = (v_img1 * spatial_weights[None, :, :])
            v_img2 = (v_img2 * spatial_weights[None, :, :])

            norm_img1 = v_img1 - v_img1.mean(dim=[1, 2])[:, None, None]
            norm_img2 = v_img2 - v_img2.mean(dim=[1, 2])[:, None, None]

            ncc1 = torch.sum(norm_img1 * norm_img2, dim=[1, 2])
            ncc2 = torch.sqrt(torch.sum(norm_img1 ** 2, dim=[1, 2]) * torch.sum(norm_img2 ** 2, dim=[1, 2]))
            ncc2 = torch.clamp_min(ncc2, 1e-6)
            ncc = 1 - ncc1 / ncc2
            return ncc

        def bilateral_ncc(v_points_c, v_normal_c, v_intrinsic, v_transformation, v_img1, v_img2,
                          v_window_size=7):
            device = v_points_c.device
            num_points = v_points_c.shape[0]
            world_up_vector = torch.zeros_like(v_normal_c)
            world_up_vector[:, 2] = 1
            right_vector = normalize_tensor(torch.cross(world_up_vector, v_normal_c))
            up_vector = normalize_tensor(torch.cross(v_normal_c, right_vector))

            height, width = v_img1.shape[:2]
            resolution = 1 / min(height, width) * torch.linalg.norm(v_points_c, dim=-1)

            index = torch.arange(v_window_size, device=device) - v_window_size // 2
            index = torch.stack(torch.meshgrid(index, index, indexing="xy"), dim=2)
            window_points = \
                (index[:, :, 0].view(1, -1, 1) * right_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3) + \
                (index[:, :, 1].view(1, -1, 1) * up_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3)
            window_points = window_points + v_points_c[:, None, None, :]

            points_2d1 = (v_intrinsic @ window_points.reshape(-1, 3).T).T
            points_2d1 = points_2d1[:, :2] / points_2d1[:, 2:3]
            points_2d2 = (v_transformation @ to_homogeneous_tensor(window_points.reshape(-1, 3)).T).T
            points_2d2 = points_2d2[:, :2] / points_2d2[:, 2:3]

            sample_imgs1 = sample_img(v_img1[None, None, :], points_2d1[None, :, :])[0]
            sample_imgs2 = sample_img(v_img2[None, None, :], points_2d2[None, :, :])[0]

            sample_imgs1 = sample_imgs1.reshape(num_points, v_window_size, v_window_size)
            sample_imgs2 = sample_imgs2.reshape(num_points, v_window_size, v_window_size)

            ncc = bilateral_ncc_(sample_imgs1, sample_imgs2)

            # Visualize
            if False:
                points_2d1 = points_2d1.reshape(num_points, v_window_size * v_window_size, 2)
                points_2d2 = points_2d2.reshape(num_points, v_window_size * v_window_size, 2)

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                point_color = (0, 255, 255)
                point_thickness = 3

                for i in range(num_points):
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)
                    img11[p_2d1[i, :, 1], p_2d1[i, :, 0]] = point_color
                    img21[p_2d2[i, :, 1], p_2d2[i, :, 0]] = point_color

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            return ncc

        # img data pre
        v_img1 = imgs[0]
        v_img2 = imgs[img_src_id + 1]

        # optimized data pre
        optimized_points_pos = optimized_distance[:, None, None] * rays_c[:, None, :]
        patches_list = graph.graph["dual_graph"].nodes  # each patch = id_vertexes
        patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]

        # polygon -> edge (using the centroid)
        # TODO：1
        edge_list = []  # num_tri * 2
        for patch in patch_vertexes_id:
            num_vertex = len(patch)
            edge_id = [[patch[i], patch[(i + 1) % num_vertex]] for i in range(num_vertex)]
            edge_list.extend(edge_id)
        num_points_per_face = torch.tensor([len(item) for item in patch_vertexes_id], dtype=torch.long, device=device)
        edge_list = torch.tensor(edge_list).to(device).to(torch.long)
        edge = optimized_points_pos[edge_list].squeeze(dim=2)
        # patch_id of edge belongs to
        edges_indices = [torch.tensor([i] * len(patch_vertexes)) for i, patch_vertexes in enumerate(patch_vertexes_id)]
        edges_indices = torch.cat(edges_indices).to(device)

        # torch.cat((scatter_mean(edge[:, 0], edges_indices, dim=0)[:, None, :].repeat_interleave(num_points_per_face,

        def initialize_plane_hypothesis(v_rays_c, v_optimized_points_pos, v_patches_list):
            # 1. calculate ncc loss of each patch
            # for each patch do:
            # a) 3d vertexes of each patch -> fitting plane
            # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
            # c) sample points on projected polygon
            # d) project sample points to img1 and img2 and get pixel
            # e) calculate the ncc loss
            poly_abcd_list = []
            ncc_list = []
            # sample_points_list = []
            # projected_vertexes_list = []
            # sample_points_2d1 = []
            # sample_points_2d2 = []
            # centroid_list = []
            for patch_idx in range(len(v_patches_list)):
                id_vertexes = v_patches_list[patch_idx]['id_vertex']
                pos_vertexes = v_optimized_points_pos[id_vertexes].view(-1, 3)
                assert (len(pos_vertexes) >= 3)
                if patch_idx == 8:
                    print("debug")
                # a) 3d vertexes of each patch -> fitting plane
                p_abcd = fit_plane_svd(pos_vertexes)
                poly_abcd_list.append(p_abcd)

                # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
                projected_points = ray_line_intersection2(p_abcd,
                                                          torch.zeros_like(v_rays_c[id_vertexes]),
                                                          v_rays_c[id_vertexes])
                # projected_points = project_points_to_plane(pos_vertexes, p_abcd)
                # projected_vertexes_list.append(projected_points)

                # c) sample points on projected polygon
                # polygon -> triangles (using the centroid)
                num_vertex = projected_points.shape[0]
                edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                centroid = torch.mean(projected_points, dim=0)
                # centroid_list.append(centroid)
                edges = projected_points[torch.tensor(edges_idx)]
                triangles = torch.cat((edges, centroid.tile(num_vertex, 1)[:, None, :]), dim=1)

                # sample points on the fitting plane
                num_samples, sample_points_on_face = \
                    sample_triangles(100, triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :])
                # sample_points_list.append(sample_points_on_face)

                triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                               triangles[:, 1, :] - triangles[:, 2, :]))
                triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                ncc = bilateral_ncc(sample_points_on_face, triangle_normal,
                                    intrinsic, transformation[img_src_id], v_img1, v_img2).mean()

                ncc_list.append(ncc)

                # Visualize
                if False:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                    p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point], 1, point_color, point_thickness)

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            # 2. sorted patches according the ncc loss
            ncc_list = torch.stack(ncc_list).view(-1)
            poly_abcd_list = torch.stack(poly_abcd_list).view(-1, 4)
            # centroid_list = torch.stack(centroid_list).view(-1, 3)
            return poly_abcd_list, ncc_list

        # For each patch, initialize plane hypothesis
        poly_abcd_list, ncc = initialize_plane_hypothesis(rays_c, optimized_points_pos, patches_list)

        def save_plane(v_abcds, rays_c, patch_vertexes_id, file_path="output/init.ply"):
            vertices = []
            polygons = []
            acc_num_vertices = 0
            for id_patch in range(len(patch_vertexes_id)):
                intersection_points = ray_line_intersection2(v_abcds[id_patch],
                                                             torch.zeros_like(rays_c[patch_vertexes_id[id_patch]]),
                                                             rays_c[patch_vertexes_id[id_patch]])
                vertices.append(intersection_points)
                polygons.append(np.arange(intersection_points.shape[0]) + acc_num_vertices)
                acc_num_vertices += intersection_points.shape[0]
                pass
            vertices = torch.cat(vertices, dim=0).cpu().numpy()
            with open(file_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex {}\nproperty float x\nproperty float y\nproperty float z\n".format(
                    acc_num_vertices))
                f.write("element face {}\nproperty list uchar int vertex_index\n".format(len(polygons)))
                f.write("end_header\n")
                for ver in vertices:
                    f.write("{} {} {}\n".format(ver[0], ver[1], ver[2]))
                for polygon in polygons:
                    f.write("{}".format(len(polygon)))
                    for item in polygon:
                        f.write(" {}".format(item))
                    f.write("\n")
                pass
            print("Save done")
            return

        def visualize_plane(ncc, patch_vertexes_id):
            for idx in range(len(patch_vertexes_id)):
                id_vertices = patch_vertexes_id[idx]
                ncc_ = ncc[idx]
                print("{:2d}: ncc={:.2f} vertex_num={:2d}".format(idx, ncc_, len(id_vertices)))

                start_points_c = optimized_points_pos[id_vertices].view(-1, 3)
                end_points_c = torch.roll(start_points_c, -1, dims=[0])

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)

                all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                all_points_c = all_points_c.reshape(-1, 3)
                p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                p_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                p_2d1 = p_2d1.reshape(-1, 2, 2)
                p_2d2 = p_2d2.reshape(-1, 2, 2)

                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(p_2d1.shape[0]):
                    cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                    cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                for i_point in range(p_2d1.shape[0]):
                    cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                cv2.imshow("1", np.concatenate([
                    np.concatenate([img11, img21], axis=1),
                    np.concatenate([img12, img22], axis=1),
                ], axis=0))
                cv2.waitKey()

        save_plane(poly_abcd_list, rays_c, patch_vertexes_id)

        visualize_plane(ncc, patch_vertexes_id)

        def sample_depth_and_angle(depth, angle, num_sample=100):
            # sample depth
            sample_depth = torch.normal(depth.repeat(num_sample, 1))
            # sample angle
            sample_angle = torch.normal(angle.repeat(num_sample, 1),
                                        torch.full_like(angle, 2 * math.pi / 6).repeat(num_sample, 1))
            # sample_angle = torch.clamp(sample_angle, min=-2 * math.pi, max=2 * math.pi)
            # add itself
            sample_depth = torch.cat((depth[None, :], sample_depth), dim=0)
            sample_angle = torch.cat((angle, sample_angle), dim=0)
            return sample_depth, sample_angle

        def compute_plane_abcd(patch_ray, ray_depth, normal):
            intersection = patch_ray * ray_depth
            d = -torch.sum(intersection * normal, dim=1, keepdim=True)
            plane_abcd = torch.cat([normal, d], dim=1)
            return intersection, plane_abcd

        def intersection_of_ray_and_all_plane(planes_abcd, vertexes_ray):
            # planes: n*4 ray: m*3
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

        def points_outside_polygon(sample_points, polygon_points, transformation_):
            sample_points = (intrinsic @ sample_points.T).T
            sample_points = (transformation_ @ to_homogeneous_tensor(sample_points).T).T
            sample_points = sample_points[:, :2] / sample_points[:, 2:3]
            polygon_points = (intrinsic @ polygon_points.T).T
            polygon_points = (transformation_ @ to_homogeneous_tensor(polygon_points).T).T
            polygon_points = polygon_points[:, :2] / polygon_points[:, 2:3]
            # 多边形的边
            poly_edges_start = polygon_points
            poly_edges_end = torch.roll(polygon_points, -1, dims=0)
            # 判断点是否在多边形外部
            cross_products = (sample_points[:, None, 0] - poly_edges_start[:, 0]) * (
                    poly_edges_end[:, 1] - poly_edges_start[:, 1]) - (
                                     sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                                     poly_edges_end[:, 0] - poly_edges_start[:, 0])
            y_diff_sign = (sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                    sample_points[:, None, 1] - poly_edges_end[:, 1])
            # 计算每个点与多边形的交点数
            intersection_count = torch.sum((cross_products > 0) & (y_diff_sign <= 0), dim=1)
            # 交点数为奇数的点在多边形内部，交点数为偶数的点在多边形外部
            outside_polygon_mask = intersection_count % 2 == 0
            return outside_polygon_mask

        def points_in_adj_patch(sample_points, v_polygon_points, transformation_):
            mask = torch.zeros(sample_points.shape[0], dtype=torch.bool).to(device)
            for polygon_points in v_polygon_points:
                mask = torch.logical_or(mask, ~points_outside_polygon(sample_points, polygon_points, transformation_))
            return mask

        def optimize_plane_hypothesis(poly_abcd_list, ncc, v_patches_list, v_dual_graph, v_c1_2_c2):
            # 1. get the intersection of the fitting plane and ray_c
            rays_c_valid_patch = np.stack([v_patches_list[i]['ray_c'] for i in range(len(v_patches_list))], axis=0)
            rays_c_valid_patch = torch.from_numpy(rays_c_valid_patch).to(device).to(torch.float32)
            # isValid, intersection = intersection_of_ray_and_plane(poly_abcd_list, rays_c_valid_patch)
            # assert (torch.sum(isValid) == isValid.shape[0])
            # depth = torch.linalg.norm(intersection, dim=-1)
            # angle = vectors_to_angles(poly_abcd_list[:, 0:3])

            sorted_values, sorted_indices = torch.sort(ncc, descending=False)

            collision_checker = Collision_checker()
            optimized_abcd_list = poly_abcd_list.clone()
            # 2. process the patches in the order of ncc loss
            for i in range(len(sorted_indices)):
                patch_id, init_ncc_loss = sorted_indices[i], sorted_values[i]
                if init_ncc_loss <= 0.1:
                    continue
                isValid, intersection = intersection_of_ray_and_plane(optimized_abcd_list[patch_id].unsqueeze(0),
                                                                      rays_c_valid_patch[patch_id].unsqueeze(0))
                assert (torch.sum(isValid) == isValid.shape[0])
                depth = torch.linalg.norm(intersection, dim=-1)
                angle = vectors_to_angles(optimized_abcd_list[patch_id, 0:3].unsqueeze(0))

                vertexes_id = torch.tensor(patch_vertexes_id[patch_id]).to(device).to(torch.long)
                vertexes_pos = optimized_points_pos[vertexes_id]
                vertexes_ray = rays_c[vertexes_id]
                patch_ray = torch.tensor(v_patches_list[int(patch_id)]['ray_c']).to(device).to(torch.float32)

                # optimize the patch
                cur_iter = 0
                ncc_loss_optimized = init_ncc_loss
                id_best = 0  # init patch will be place at 0 index
                num_plane_sample = 100
                while True:
                    # 1. Sample candidate plane
                    samples_depth, samples_angle = sample_depth_and_angle(depth, angle, num_plane_sample)
                    samples_abc = angles_to_vectors(samples_angle)
                    samples_planes_intersection, samples_abcd = compute_plane_abcd(patch_ray, samples_depth,
                                                                                   samples_abc)

                    # num_samples_plane * num_vertexes_cur_patch * 3
                    samples_planes_vertexes_pos = intersection_of_ray_and_all_plane(samples_abcd, vertexes_ray)

                    # 2. Get intersection points
                    # candidate_triangle_points (N, 3, 3)
                    num_vertex = vertexes_id.shape[0]
                    edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                    centroid = torch.mean(samples_planes_vertexes_pos, dim=1)
                    # centroid_list.append(centroid)
                    edges = samples_planes_vertexes_pos[:, torch.tensor(edges_idx)]
                    triangles = torch.cat((edges, centroid[:, None, None, :].tile(1, num_vertex, 1, 1)), dim=2)
                    triangles = triangles.view(-1, 3, 3)  # (101*num_tri,3,3)

                    # 3. calculate loss (point ncc loss and line loss)
                    # sample points on the fitting plane
                    # get each sample point's ncc loss, then aggregate to each patch
                    num_samples, sample_points_on_face = sample_triangles(100, triangles[:, 0, :],
                                                                          triangles[:, 1, :], triangles[:, 2, :])

                    triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                                   triangles[:, 1, :] - triangles[:, 2, :]))
                    triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                    ncc_loss_points = bilateral_ncc(sample_points_on_face, triangle_normal, intrinsic,
                                                    transformation[img_src_id], v_img1, v_img2)

                    # get the edge point of each sample patch plane
                    ncc_loss_edge, ncc_loss_mask, num_samples_per_edge = evaluate_candidates(edges.reshape(-1, 2, 3),
                                                                                             intrinsic,
                                                                                             transformation[img_src_id],
                                                                                             v_img1, v_img2)
                    ncc_loss_edge = ncc_loss_edge.view(num_plane_sample + 1, -1)
                    ncc_loss_edge = torch.mean(ncc_loss_edge, dim=1)

                    # filter the points int adj patch (in img2 candidate)
                    filter_points = True
                    if filter_points:
                        sample_points_on_face_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            sample_points_on_face).transpose(0, 1)).transpose(0, 1)[:, :3]
                        point_mask = collision_checker.check_ray(
                            torch.zeros_like(sample_points_on_face_c2),
                            sample_points_on_face_c2)
                        edges_shape = edges.shape
                        edges_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            edges.reshape(-1, 3)).transpose(0, 1)).transpose(0, 1)[:, :3]
                        edge_mask = collision_checker.check_ray(
                            torch.zeros_like(edges_c2),
                            edges_c2).reshape(*edges_shape[:-1])
                        edge_mask = torch.all(edge_mask, dim=-1)

                        # adj_patch_id_list = [edge[1] for edge in list(v_dual_graph.edges(int(patch_id)))]
                        # adj_patch_vertex_pos = []
                        # for adj_patch_id in adj_patch_id_list:
                        #     vertexes_id_adj = torch.tensor(patch_vertexes_id[adj_patch_id]).to(device).to(torch.long)
                        #     vertexes_ray_adj = rays_c[vertexes_id_adj]
                        #     valid_mask, vertexes_pos_adj = intersection_of_ray_and_plane(
                        #         poly_abcd_list[patch_id].tile(vertexes_ray_adj.shape[0], 1),
                        #         vertexes_ray_adj)
                        #     adj_patch_vertex_pos.append(vertexes_pos_adj)
                        # mask_points_in_adj_patch = points_in_adj_patch(sample_points_on_face,
                        #                                                adj_patch_vertex_pos,
                        #                                                transformation[0])
                        # ncc_loss_points = ncc_loss_points[~mask_points_in_adj_patch]
                        ncc_loss_points = ncc_loss_points[~point_mask]
                        num_samples = scatter_add((~point_mask).to(torch.int64),
                                                  torch.arange(num_samples.shape[0], device=device)
                                                  .repeat_interleave(num_samples))

                    ncc_loss_edge = torch.mean(ncc_loss_edge.view(num_plane_sample + 1, -1), dim=1)
                    # 5. Find best
                    points_to_tri = torch.arange(num_samples.shape[0]).to(device).repeat_interleave(num_samples)
                    ncc_loss_triangle = scatter_mean(ncc_loss_points, points_to_tri, dim=0)
                    if filter_points:
                        pass
                        # points_to_tri = points_to_tri[~mask_points_in_adj_patch]
                        # tri_idx = torch.arange(num_samples.shape[0]).to(device)
                        # missing_tri_idx = tri_idx[~torch.isin(tri_idx, torch.unique(points_to_tri))]
                        # ncc_loss_triangle[missing_tri_idx] = 0
                    ncc_loss_triangle = ncc_loss_triangle.view(num_plane_sample + 1, -1)
                    ncc_loss_patch = torch.mean(ncc_loss_triangle, dim=1)
                    ncc_loss_patch = ncc_loss_patch + ncc_loss_edge
                    id_best = torch.argmin(ncc_loss_patch, dim=-1)
                    best_abcd = samples_abcd[id_best]
                    # update
                    optimized_abcd_list[patch_id] = best_abcd
                    ncc_loss_optimized = ncc_loss_patch[id_best]
                    depth = torch.linalg.norm(samples_planes_intersection[id_best].unsqueeze(0), dim=-1)
                    angle = vectors_to_angles(best_abcd[0:3].unsqueeze(0))

                    # 6. Detect neighbour
                    num_sample_points = num_samples.view(num_plane_sample + 1, -1).sum(dim=-1)
                    num_sample_points = torch.cumsum(num_sample_points, dim=0)
                    if id_best == 0:
                        sample_points_c = sample_points_on_face[0:num_sample_points[id_best]]
                    else:
                        sample_points_c = sample_points_on_face[
                                          num_sample_points[id_best - 1]:num_sample_points[id_best]]
                    all_points_c = samples_planes_vertexes_pos[id_best].view(-1, 3)

                    def vis_(all_points_c_, sample_points_c_, patch_id_, cur_iter_):
                        img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img21 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)
                        img22 = cv2.cvtColor(src_imgs[img_src_id], cv2.COLOR_GRAY2BGR)

                        shape = img11.shape[:2][::-1]
                        # vertex
                        p_2d1 = (intrinsic @ all_points_c_.T).T.cpu().numpy()
                        p_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(all_points_c_).T).T.cpu().numpy()
                        p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                        p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
                        p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                        p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                        # sample points
                        s_2d1 = (intrinsic @ sample_points_c_.T).T.cpu().numpy()
                        s_2d2 = (transformation[img_src_id] @ to_homogeneous_tensor(sample_points_c_).T).T.cpu().numpy()
                        s_2d1 = s_2d1[:, :2] / s_2d1[:, 2:3]
                        s_2d2 = s_2d2[:, :2] / s_2d2[:, 2:3]
                        s_2d1 = np.around(s_2d1 * shape).astype(np.int64)
                        s_2d2 = np.around(s_2d2 * shape).astype(np.int64)

                        line_color = (0, 0, 255)
                        line_thickness = 2
                        vertex_point_color = (0, 0, 255)
                        sample_point_color = (0, 255, 0)
                        point_thickness = 3

                        for i_point in range(s_2d1.shape[0]):
                            cv2.circle(img11, s_2d1[i_point], 1, sample_point_color, 2)
                            cv2.circle(img21, s_2d2[i_point], 1, sample_point_color, 2)

                        for i_point in range(p_2d1.shape[0]):
                            cv2.circle(img11, p_2d1[i_point], 1, vertex_point_color, point_thickness)
                            cv2.circle(img21, p_2d2[i_point], 1, vertex_point_color, point_thickness)

                        cv2.imwrite(os.path.join(v_log_root, "patch_{}_iter_{}.jpg").format(patch_id_, cur_iter_),
                                    np.concatenate([
                                        np.concatenate([img11, img21], axis=1),
                                        np.concatenate([img12, img22], axis=1),
                                    ], axis=0)
                                    )

                    if cur_iter == 0:
                        all_points_c_init = samples_planes_vertexes_pos[0].view(-1, 3)
                        sample_points_c_init = sample_points_on_face[0:num_sample_points[0]]
                        vis_(all_points_c_init, sample_points_c_init, patch_id, cur_iter)

                    vis_(all_points_c, sample_points_c, patch_id, cur_iter + 1)

                    if ncc_loss_optimized <= 0.01 or cur_iter >= 5:
                        print(
                            "End: patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        break
                    else:
                        print("patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        cur_iter = cur_iter + 1

                # Update triangles
                final_edge_points = all_points_c[torch.tensor(edges_idx)]
                final_centroid = torch.mean(all_points_c, dim=0)
                final_triangles = torch.cat(
                    (final_edge_points, final_centroid[None, None, :].tile(num_vertex, 1, 1)), dim=1)

                final_triangles_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                    final_triangles).transpose(1, 2)).transpose(1, 2)[:, :, :3]
                collision_checker.add_triangles(final_triangles_c2)
            return optimized_abcd_list

        # Optimize plane hypothesis based on: 1) NCC; 2) Distance to the optimized vertices
        optimized_abcd_list = optimize_plane_hypothesis(poly_abcd_list, ncc, patches_list, graph.graph["dual_graph"],
                                                        c1_2_c2)

        save_plane(optimized_abcd_list, rays_c, patch_vertexes_id, file_path="output/optimized.ply")
        return
        # Determine the final location of vertices
        determine_vertices_location()

        exit()

