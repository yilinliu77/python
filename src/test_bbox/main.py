import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

base_dir = r'D:\DATASET\KITTI\3DDetection\data_object_image_2'
use_pitch = None


class cfg():
    def __init__(self):
        # Todo: set base_dir to kitti/image_2
        self.base_dir = base_dir

        # Todo: set the base network: vgg16, vgg16_conv or mobilenet_v2
        self.network = 'mobilenet_v2'

        # set the bin size
        self.bin = 2

        # set the train/val split
        self.split = 0.8

        # set overlapping
        self.overlap = 0.1

        # set jittered
        self.jit = 3

        # set the normalized image size
        self.norm_w = 224
        self.norm_h = 224

        # set the batch size
        self.batch_size = 8

        # set the categories
        self.KITTI_cat = ['Car', 'Cyclist', 'Pedestrian']
        # self.KITTI_cat = ['Car']


class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2 * np.pi - div < rot_local < 2 * np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi


class ReadDir():
    def __init__(self,
                 base_dir,
                 subset='training',
                 tracklet_date='2011_09_26',
                 tracklet_file='2011_09_26_drive_0084_sync'
                 ):
        # Todo: set local base dir
        # self.base_dir = '/home/user/PycharmProjects/dataset/kitti/data_object_image_2/'
        self.base_dir = base_dir

        # if use kitti training data for train/val evaluation
        if subset == 'training':
            self.label_dir = os.path.join(self.base_dir, subset, 'label_2/')
            self.image_dir = os.path.join(self.base_dir, subset, 'image_2/')
            self.calib_dir = os.path.join(self.base_dir, subset, 'calib/')
            self.prediction_dir = os.path.join(self.base_dir, subset, 'box_3d/')

        # if use raw data
        if subset == 'tracklet':
            self.tracklet_drive = os.path.join(self.base_dir, tracklet_date, tracklet_file)
            self.label_dir = os.path.join(self.tracklet_drive, 'label_02/')
            self.image_dir = os.path.join(self.tracklet_drive, 'image_02/data/')
            self.calib_dir = os.path.join(self.tracklet_drive, 'calib_02/')
            self.prediction_dir = os.path.join(self.tracklet_drive, 'box_3d_mobilenet/')


def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape / 2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0, :]))


def draw_birdeyes(ax2, line_gt, line_p, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
    ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)


def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    if use_pitch:
        pass
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D


def draw_3Dbox(ax, P2, line, color):
    corners_2D = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)


def visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES):
    for index in range(start_frame, end_frame):
        image_file = os.path.join(image_path, dataset[index] + '.png')
        label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index] + '.txt')
        calibration_file = os.path.join(calib_path, dataset[index] + '.txt')
        for line in open(calibration_file):
            if 'P2' in line:
                P2 = line.split(' ')
                P2 = np.asarray([float(i) for i in P2[1:]])
                P2 = np.reshape(P2, (3, 4))

        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        # fig.tight_layout()
        gs = GridSpec(1, 4)
        gs.update(wspace=0)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])

        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
        image = Image.open(image_file).convert('RGB')
        shape = 900
        birdimage = np.zeros((shape, shape, 3), np.uint8)

        # with open(label_file) as f1, open(prediction_file) as f2:
        with open(label_file) as f1:
            # for line_gt, line_p in zip(f1, f2):
            for line_gt in f1:
                line_gt = line_gt.strip().split(' ')
                # line_p = line_p.strip().split(' ')

                truncated = np.abs(float(line_gt[1]))
                occluded = np.abs(float(line_gt[2]))
                trunc_level = 1 if args.a == 'training' else 255

                # truncated object in dataset is not observable
                if line_gt[0] in VEHICLES and truncated < trunc_level:
                    color = 'green'
                    if line_gt[0] == 'Cyclist':
                        color = 'yellow'
                    elif line_gt[0] == 'Pedestrian':
                        color = 'cyan'
                    draw_3Dbox(ax, P2, line_gt, color)
                    draw_birdeyes(ax2, line_gt, line_gt, shape)

        # visualize 3D bounding box
        ax.imshow(image)
        ax.set_xticks([])  # remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

        # visualize bird eye view
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # add legend
        handles, labels = ax2.get_legend_handles_labels()
        legend = ax2.legend([handles[0], handles[1]], [labels[0], labels[1]], loc='lower right',
                            fontsize='x-small', framealpha=0.2)
        for text in legend.get_texts():
            plt.setp(text, color='w')

        print(dataset[index])
        if args.save == False:
            plt.show()
        else:
            fig.savefig(os.path.join(args.path, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        # video_writer.write(np.uint8(fig))


def main(args):
    dir = ReadDir(base_dir=base_dir, subset=args.a,
                  tracklet_date='2011_09_26', tracklet_file='2011_09_26_drive_0093_sync')
    label_path = dir.label_dir
    image_path = dir.image_dir
    calib_path = dir.calib_dir
    pred_path = dir.prediction_dir

    dataset = [name.split('.')[0] for name in sorted(os.listdir(image_path))]

    VEHICLES = cfg().KITTI_cat

    visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES)


if __name__ == '__main__':
    start_frame = 0
    end_frame = 432

    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default='../result_mobilenet_0093', help='Output Image folder')
    parser.add_argument('--pitch', type=bool, default=False, help='whether if pitch is involved')
    args = parser.parse_args()
    use_pitch=args.pitch
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    main(args)
