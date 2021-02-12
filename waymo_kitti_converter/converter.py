from avcv.utils import multi_thread
import mmcv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from multiprocessing import Pool
from os.path import join, isdir
import argparse
from glob import glob

from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
import os
import sys
sys.path.insert(0, './tools')
import kitti_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Abbreviations:
# WOD: Waymo Open Dataset
# FOV: field of view
# SDC: self-driving car
# 3dbox: 3D bounding box

# Some 3D bounding boxes do not contain any points
# This switch, when set True, filters these boxes
# It is safe to filter these boxes because they are not counted towards evaluation anyway
filter_empty_3dboxes = False


# There is no bounding box annotations in the No Label Zone (NLZ)
# if set True, points in the NLZ are filtered
filter_no_label_zone_points = True

# Only bounding boxes of certain classes are converted
# Note: Waymo Open Dataset evaluates for ALL_NS, including only 'VEHICLE', 'PEDESTRIAN', 'CYCLIST'
selected_waymo_classes = [
    # 'UNKNOWN',
    'VEHICLE',
    'PEDESTRIAN',
    # 'SIGN',
    'CYCLIST'
]

CAMERA_NAME2ID = {
        'FRONT':1,
    'FRONT_LEFT':2,
    'FRONT_RIGHT': 3,
    'SIDE_LEFT':4,
    'SIDE_RIGHT':5,
}
CAMERA_ID2NAME = {
        1:'FRONT',
    2:'FRONT_LEFT',
    3:'FRONT_RIGHT',
    4:'SIDE_LEFT',
    5:'SIDE_RIGHT',
}


# Only data collected in specific locations will be converted
# If set None, this filter is disabled (all data will thus be converted)
# Available options: location_sf (main dataset)
selected_waymo_locations = None

# Save track id
save_track_id = True

# DATA_PATH = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/domain_adaptation_training_labelled(partial)'
# KITTI_PATH = '/home/alex/github/waymo_to_kitti_converter/tools/pose'


class WaymoToKITTI(object):

    def __init__(self, load_dir, save_dir, prefix, num_proc, camera_names, max_record=None, convert_lidar=False, convert_pose=False):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()
        self.camera_names = camera_names
        self.lidar_list = ['_FRONT', '_FRONT_RIGHT',
                           '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.type_list = ['UNKNOWN', 'VEHICLE',
                          'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }
        self.convert_lidar = convert_lidar
        self.convert_pose = convert_pose
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.num_proc = int(num_proc)

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))
        if max_record is not None:
            self.tfrecord_pathnames = self.tfrecord_pathnames[:max_record]

        self.label_save_dir = self.save_dir + '/label_'
        self.label_all_save_dir = self.save_dir + '/label_all'
        self.image_save_dir = self.save_dir + '/image_'
        self.calib_save_dir = self.save_dir + '/calib'
        self.point_cloud_save_dir = self.save_dir + '/velodyne'
        self.pose_save_dir = self.save_dir + '/pose'
        self.axes_transformation = np.array([
            [0,-1,0,0],
            [0,0,-1,0],
            [1,0,0,0],
            [0,0,0,1]])
        self.create_folder()

    def check_file_exists(self, file_path):
        return os.path.exists(file_path) and self.num_proc != 1

    def convert(self):
        print("start converting ...")
        if self.num_proc == 1:
            for i in range(len(self)):
                print('Convertting :', i)
                self.convert_one(i)
        else:
            with Pool(self.num_proc) as p:
                r = list(tqdm.tqdm(p.imap(self.convert_one,
                                          range(len(self))), total=len(self)))
        # print("\nfinished ...")

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        def f(input):
            frame_idx, data = input

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if selected_waymo_locations is not None and frame.context.stats.location not in selected_waymo_locations:
                return

            # save images
            self.save_image(frame, file_idx, frame_idx)
            # parse label files
            self.save_label_and_calib(frame, file_idx, frame_idx)
            if self.convert_lidar:
                # parse point clouds
                self.save_lidar(frame, file_idx, frame_idx)
            if self.convert_pose:
                # parse pose files
                self.save_pose(frame, file_idx, frame_idx)

        inputs = []
        for frame_idx, data in enumerate(dataset):
            inputs.append((frame_idx, data))

        multi_thread(f, inputs, verbose=True, max_workers=min(4, self.num_proc), desc=f'File index {file_idx}')

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """ parse and save the images in png format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        self.image_shapes = dict()
        for img in frame.images:
            camera_id = img.name-1
            if  img.name in self.camera_names:
                img_path = self.image_save_dir + str(camera_id) + '/' + self.prefix + str(
                    file_idx).zfill(3) + str(frame_idx).zfill(3) + '.jpg'
                if not self.check_file_exists(img_path):
                    img = cv2.imdecode(np.frombuffer(
                        img.image, np.uint8), cv2.IMREAD_COLOR)
                    self.image_shapes[camera_id+1] = (img.shape[0], img.shape[1])
                    mmcv.imwrite(img, img_path)
                else:
                    self.image_shapes[camera_id+1] = img_path

    def save_calib(self, frame, file_idx, frame_idx):
        """ parse and save the calibration data
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        calib_context = ''

        self.cam_intrinsics = dict()
        self.cam_extrinsics = dict()
        for camera in frame.context.camera_calibrations:
            # extrinsics
            _T_cam_to_velo = np.array(camera.extrinsic.transform).reshape(4, 4)
            T_velo_to_cam = np.linalg.inv(_T_cam_to_velo)

            self.cam_extrinsics[camera.name] = T_velo_to_cam.copy()
            # intrinsics
            cam_intrinsic = np.zeros((3, 4))
            cam_intrinsic[0, 0] = camera.intrinsic[0]
            cam_intrinsic[1, 1] = camera.intrinsic[1]
            cam_intrinsic[0, 2] = camera.intrinsic[2]
            cam_intrinsic[1, 2] = camera.intrinsic[3]
            cam_intrinsic[2, 2] = 1
            self.cam_intrinsics[camera.name] = cam_intrinsic.copy()

        for camera_name, P in self.cam_intrinsics.items():
            P = P.reshape(12)
            calib_context += f"P{camera_name}: " + \
                " ".join(['{}'.format(i) for i in P]) + '\n'

        calib_context += "R0_rect" + ": " + \
            " ".join(['{}'.format(i)
                for i in np.eye(3).astype(np.float32).flatten()]) + '\n'

        for camera_name, T_velo_to_cam in self.cam_extrinsics.items():
            Tr_velo_to_cam = self.axes_transformation @ T_velo_to_cam
            calib_context += f"Tr_velo_to_cam_{camera_name}" + ": " + \
                " ".join(['{}'.format(i)for i in Tr_velo_to_cam[:3, :].reshape(12)]) + '\n'


        #----------------- SAVE
        save_path = self.calib_save_dir + '/' + self.prefix + \
            str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt'
        if not os.path.exists(save_path):
            with open(save_path, 'w+') as fp_calib:
                fp_calib.write(calib_context)

    def save_lidar(self, frame, file_idx, frame_idx):
        """ parse and save the lidar data in psd format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
                """
        pc_path = self.point_cloud_save_dir + '/' + self.prefix + \
            str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.bin'
        if not self.check_file_exists(pc_path):

            range_images, camera_projections, range_image_top_pose = parse_range_image_and_camera_projection(
                frame)
            points_0, cp_points_0, intensity_0 = self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
            points_0 = np.concatenate(points_0, axis=0)
            intensity_0 = np.concatenate(intensity_0, axis=0)

            points_1, cp_points_1, intensity_1 = self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
            points_1 = np.concatenate(points_1, axis=0)
            intensity_1 = np.concatenate(intensity_1, axis=0)

            points = np.concatenate([points_0, points_1], axis=0)
            # print('points_0', points_0.shape, 'points_1', points_1.shape, 'points', points.shape)
            intensity = np.concatenate([intensity_0, intensity_1], axis=0)
            # points = points_1
            # intensity = intensity_1

            # reference frame:
            # front-left-up (waymo) -> right-down-front(kitti)
            # lidar frame:
            # ?-?-up (waymo) -> front-right-up (kitti)

            # print('bef\n', points)
            # print('bef\n', points.dtype)
            # points = np.transpose(points)  # (n, 3) -> (3, n)
            # tf = np.array([
            #     [0.0, -1.0,  0.0],
            #     [0.0,  0.0, -1.0],
            #     [1.0,  0.0,  0.0]
            # ])
            # points = np.matmul(tf, points)
            # points = np.transpose(points)  # (3, n) -> (n, 3)
            # print('aft\n', points)
            # print('aft\n', points.dtype)

            # concatenate x,y,z and intensity
            point_cloud = np.column_stack((points, intensity))

            # print(point_cloud.shape)

            # save
            # note: must save as float32, otherwise loading errors
            point_cloud.astype(np.float32).tofile(pc_path)

    def save_label_and_calib(self, frame, file_idx, frame_idx):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
                """
        label_path = self.label_all_save_dir + '/' + self.prefix + \
            str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt'
        if self.check_file_exists(label_path):
            return

        self.save_calib(frame, file_idx, frame_idx)

        fp_label_all = open(label_path, 'w+')
        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # waymo: bounding box origin is at the center
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        fp_labels = dict()
        for name in range(5):
            name = str(name)
            fp_labels[name] = open(self.label_save_dir + name + '/' + self.prefix +
                            str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'a')

        # print([i.type for i in frame.laser_labels])
        for name in range(5):
            name = str(name)
            cam_id = int(name)+1
            for obj in frame.laser_labels:
                # calculate bounding box
                id = obj.id
                # for lidar in self.lidar_list:
                #     if id + lidar in id_to_bbox:
                #         bounding_box = id_to_bbox.get(id + lidar)
                #         name = str(id_to_name.get(id + lidar))
                #         break


                my_type = self.type_list[obj.type]

                if my_type not in selected_waymo_classes:
                    continue

                if filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                    continue

                bounding_box = (0, 0, 0, 0)
                my_type = self.waymo_to_kitti_class_map[my_type]
                height = obj.box.height  # up/down
                width = obj.box.width  # left/right
                length = obj.box.length  # front/back

                x = obj.box.center_x
                y = obj.box.center_y
                z = obj.box.center_z - height / 2
                # print('bef', x,y,z)
                # project bounding box to the virtual reference frame
                camera_id = int(name)+1#CAMERA_NAME2ID #[lidar[1:]]



                rotation_y = -obj.box.heading - np.pi / 2 
                if name == '1':
                    rotation_y+=np.pi/4
                elif name == '2':
                    rotation_y-=np.pi/4
                elif name == '3':
                    rotation_y+=np.pi/2
                elif name == '4':
                    rotation_y-=np.pi/2

                pt_ref = self.axes_transformation @ self.cam_extrinsics[camera_id] @ np.array([x, y, z, 1]).reshape((4, 1))

                x, y, z, _ = pt_ref.flatten().tolist()
                if z <= 0: continue
                box_3d_on_image = self.compute_box_3d(length, width, height, x,y,z, rotation_y, self.cam_intrinsics[cam_id])
                if not kitti_util.is_in_image(box_3d_on_image, self.image_shapes[cam_id]):
                    continue


                track_id = obj.id
                # not available
                truncated = 0
                occluded = 0
                # alpha:
                # we set alpha to the default -10, the same as nuscenes to kitti tool
                # contribution is welcome
                alpha = -10


                line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                    occluded,
                                                                                    round(
                                                                                        alpha, 2),
                                                                                    round(
                                                                                        bounding_box[0], 2),
                                                                                    round(
                                                                                        bounding_box[1], 2),
                                                                                    round(
                                                                                        bounding_box[2], 2),
                                                                                    round(
                                                                                        bounding_box[3], 2),
                                                                                    round(
                                                                                        height, 2),
                                                                                    round(
                                                                                        width, 2),
                                                                                    round(
                                                                                        length, 2),
                                                                                    round(
                                                                                        x, 2),
                                                                                    round(
                                                                                        y, 2),
                                                                                    round(
                                                                                        z, 2),
                                                                                    round(rotation_y, 2))
                if save_track_id:
                    line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
                else:
                    line_all = line[:-1] + ' ' + name + '\n'
                # store the label
                fp_labels[name].write(line)

                fp_label_all.write(line_all)

        for fp_label in fp_labels.values():
            fp_label.close()

        fp_label_all.close()

        # print(file_idx, frame_idx)


    def save_pose(self, frame, file_idx, frame_idx):
        """ Save self driving car (SDC)'s own pose

        Note that SDC's own pose is not included in the regular training of KITTI dataset
        KITTI raw dataset contains ego motion files but are not often used
        Pose is important for algorithms that takes advantage of the temporal information

        """

        pose_path = join(self.pose_save_dir, self.prefix +
                         str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt')
        if self.check_file_exists(pose_path):
            return
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(pose_path, pose)

    def create_folder(self):
        for d in [self.label_all_save_dir, self.calib_save_dir, self.point_cloud_save_dir, self.pose_save_dir]:
            if not isdir(d):
                os.makedirs(d)
        for d in [self.label_save_dir, self.image_save_dir]:
            for i in range(5):
                if not isdir(d + str(i)):
                    os.makedirs(d + str(i))

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
             camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[...,
                                        0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min,
                                 c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            # No Label Zone
            if filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                # print(range_image_tensor[range_image_tensor[..., 3] == 1.0])
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(
                    value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(
                value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor,
                                            tf.where(range_image_mask))
            intensity.append(intensity_tensor.numpy()[:, 1])

        return points, cp_points, intensity

    # def get_intensity(self, frame, range_images, ri_index=0):
    #     """Convert range images to point cloud.
    #     Args:
    #       frame: open dataset frame
    #        range_images: A dict of {laser_name,
    #          [range_image_first_return, range_image_second_return]}.
    #        camera_projections: A dict of {laser_name,
    #          [camera_projection_from_first_return,
    #           camera_projection_from_second_return]}.
    #       range_image_top_pose: range image pixel pose for top lidar.
    #       ri_index: 0 for the first return, 1 for the second return.
    #     Returns:
    #       intensity: {[N, 1]} list of intensity of length 5 (number of lidars).
    #     """
    #     calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    #     intensity = []
    #     for c in calibrations:
    #         range_image = range_images[c.name][ri_index]
    #         range_image_tensor = tf.reshape(
    #             tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    #         range_image_mask = range_image_tensor[..., 0] > 0
    #         intensity_tensor = tf.gather_nd(range_image_tensor,
    #                                         tf.where(range_image_mask))
    #         intensity.append(intensity_tensor.numpy()[:, 1])
    #
    #     return intensity

    def compute_box_3d(self, l,w,h,tx,ty,tz, ry, P):
        """ Takes an object and a projection matrix (P) and projects the 3d
            bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in in rect camera coord.
        """
        R = kitti_util.roty(ry)

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + tx
        corners_3d[1, :] = corners_3d[1, :] + ty
        corners_3d[2, :] = corners_3d[2, :] + tz
        # if np.any(corners_3d[2, :] < 0.1):
        #     corners_2d = None
        #     return corners_2d, np.transpose(corners_3d)

        # # project the 3d bounding box into the image plane
        corners_2d = kitti_util.project_to_image(np.transpose(corners_3d), P)[0]
        return corners_2d


    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument(
        'save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='',
                        help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1,
                        help='Number of processes to spawn')
    parser.add_argument('--max-record', default=None,
                        type=int, help='Number of processes to spawn')
    parser.add_argument('--camera_names', default='1,2,3,4,5',
                        type=str, help='Camera ids ')
    parser.add_argument('--convert_lidar', action='store_true',
                        default=False, help='Convert lidar is time-consuming')
    parser.add_argument('--convert_pose', action='store_true',
                        default=False, help='Convert pose is time-consuming')

    args = parser.parse_args()
    cameras = [int(i) for i in args.camera_names.split(',')]

    converter = WaymoToKITTI(args.load_dir, args.save_dir, args.prefix, args.num_proc, max_record=args.max_record,
                             camera_names=cameras, convert_lidar=args.convert_lidar, convert_pose=args.convert_pose)
    converter.convert()
