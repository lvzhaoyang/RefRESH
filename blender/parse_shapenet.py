"""
MIT License

Copyright (c) 2018 Zhaoyang Lv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys, os, time, argparse, random
import io_utils
import numpy as np
import os.path as osp

from pickle import load, dump
from os.path import join, dirname, realpath, exists
from scipy.misc import imread

import OpenEXR, Imath
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

sys.path.insert(0, ".")
from geometry import depth2flow

class ShapeNetSceneParser:

    def __init__(self, post_fix = ''):

        self.params = io_utils.load_file('configs/shapenet_config', 'SHAPENET_MODELS')

        self.output_path = osp.join(self.params['output_path'], post_fix)
        self.tmp_path = osp.join(self.params['tmp_path'], post_fix)

    def run(self):

        print('generate output for {:s}'.format(self.output_path))

        info = {'color':         [],
                'depth':         [],
                'pose':          [],
                'invalid':       [],
                'object_mask':   [],
                'object_2D_box': {},
                # whether the object is actually valid appearing in the current frame
                'object_visible':{}, # the binary label for each frame
                'object_visible_frames': {} # the frame index overall
        }

        K = self.params['pinhole']
        info['calib'] = [K['fx'], K['fy'], K['cx'], K['cy']]

        with open(osp.join(self.tmp_path, 'info.pkl'), 'rb') as f:
            files = load(f)
            info['pose'] = files['camera_pose']
            info['object_poses'] = files['object_pose']
            info['object_3D_box'] = files['object_3D_box']
            info['object_poses'] = files['object_pose']

        color_dir    = join(self.output_path, 'color')
        depth_dir    = join(self.output_path, 'depth')
        instance_dir = join(self.output_path, 'instance')
        invalid_dir  = join(self.output_path, 'invalid')

        obj_name_list = list(files['object_pose'].keys())
        for obj_name in obj_name_list:
            info['object_2D_box'][obj_name] = []
            info['object_visible'][obj_name]= []
            info['object_visible_frames'][obj_name] = []

        for idx in range(0, self.params['trajectory']['views']):

            exr_file = osp.join(self.tmp_path, 'Image{:04d}.exr'.format(idx))
            exr = OpenEXR.InputFile(exr_file)
            size = (self.params['height'], self.params['width'])

            invalid_mask = np.zeros(size, np.uint8)

            output_name = str(idx).zfill(6)
            print('generate file: {:}'.format(output_name))
            filename_png = output_name+'.png'

            # process rendered image
            color = self.__read_color(exr, size)
            rendered_color_file = osp.abspath(join(color_dir,filename_png))
            io_utils.image_write(rendered_color_file, color)
            info['color'].append(rendered_color_file)

            if self.params['output_types']['depth']:
                depth = self.__read_depth(exr, size)
                depth_file = osp.abspath(join(depth_dir, filename_png))
                io_utils.pngdepth_write(depth_file, depth)
                info['depth'].append(depth_file)

            if self.params['output_types']['segm']: # instance segmentation
                obj_mask = self.__read_object_mask(exr, size)
                for obj_name in obj_name_list:
                    obj_index = int(obj_name.split('_')[-1])
                    obj_box2D, obj_valid = self.__get_bounding_box(obj_mask, obj_index)
                    info['object_2D_box'][obj_name].append(obj_box2D)
                    info['object_visible'][obj_name].append(obj_valid)
                    if obj_valid > 0:
                        info['object_visible_frames'][obj_name].append(idx)
                instance_file = osp.abspath(join(instance_dir, filename_png))
                io_utils.pnginstance_write(instance_file, obj_mask)
                info['object_mask'].append(instance_file)

            # if self.params['output_types']['gtflow']:
            #     forward_flow, backward_flow = self.__read_flow(exr, size)
            # import matplotlib.pyplot as plt
            # for obj_name, obj_poses in object_poses.items():
            #     cam_pose_this = info['pose'][idx]
            #     cam_pose_next = info['pose'][idx+1]
            #     obj_pose_this = obj_poses[idx]
            #     obj_pose_next = obj_poses[idx+1]
            #
            #     import functools
            #     transform = functools.reduce(np.dot,
            #     [np.linalg.inv(cam_pose_next), obj_pose_next,
            #      np.linalg.inv(obj_pose_this), cam_pose_this])
            #
            #     # get the correct transform between the camrea and the pose
            #     # sanity check of the object poses
            #     K_mat = np.eye(4)
            #     K_mat[0,0], K_mat[1,1], K_mat[0,2], K_mat[1,2] = K['fx'], K['fy'], K['cx'], K['cy']
            #     object_flow = depth2flow(depth, K_mat, transform)
            #     flow_diff_ratio = np.abs((object_flow - forward_flow)/object_flow)
            #
            #     print(obj_name)
            #     plt.figure()
            #     plt.imshow(object_flow[:,:,0])
            #     plt.figure()
            #     plt.imshow(np.clip(flow_diff_ratio[:,:,0], -400, 400))
            #     plt.figure()
            #     plt.imshow(np.clip(flow_diff_ratio[:,:,1], -400, 400))
            #     plt.show()

            invalid_mask_file = osp.abspath(join(invalid_dir, filename_png))
            io_utils.image_write(invalid_mask_file, invalid_mask)
            info['invalid'].append(invalid_mask_file)

        # if len(info['object_visible'][obj_name]) < 1:
        #     import pdb; pdb.set_trace()
        #
        dataset_path = osp.join(self.output_path, 'info.pkl')
        with open(dataset_path, 'wb') as output:
            dump(info, output)

    def __read_flow(self, exr, size):
        """ Read the forward flow and backward flow from the exr file
        """
        forward_u = -np.reshape(np.fromstring(exr.channel('RenderLayer.Vector.Z', FLOAT), dtype=np.float32), size)
        forward_v =  np.reshape(np.fromstring(exr.channel('RenderLayer.Vector.W', FLOAT), dtype=np.float32), size)
        forward_flow = np.stack((forward_u, forward_v),axis=2)

        backward_u =  np.reshape(np.fromstring(exr.channel('RenderLayer.Vector.X', FLOAT), dtype=np.float32), size)
        backward_v = -np.reshape(np.fromstring(exr.channel('RenderLayer.Vector.Y', FLOAT), dtype=np.float32), size)
        backward_flow = np.stack((backward_u, backward_v),axis=2)

        return forward_flow, backward_flow

    def __read_depth(self, exr, size):
        """ Read depth from the exr file
        """
        depth = np.reshape(np.fromstring(exr.channel('RenderLayer.Depth.Z', FLOAT), dtype=np.float32), size)
        return depth

    def __read_color(self, exr, size):
        """ Read rendered color image from the exr file
        """
        cc_r = np.fromstring(exr.channel('RenderLayer.Combined.R', FLOAT), dtype=np.float32)
        cc_g = np.fromstring(exr.channel('RenderLayer.Combined.G', FLOAT), dtype=np.float32)
        cc_b = np.fromstring(exr.channel('RenderLayer.Combined.B', FLOAT), dtype=np.float32)
        cc_a = np.fromstring(exr.channel('RenderLayer.Combined.A', FLOAT), dtype=np.float32)

        cc_r = np.reshape((cc_r * 255 / np.max(cc_r)).astype('uint8'), size)
        cc_g = np.reshape((cc_g * 255 / np.max(cc_g)).astype('uint8'), size)
        cc_b = np.reshape((cc_b * 255 / np.max(cc_b)).astype('uint8'), size)
        cc_a = np.reshape((cc_a * 255 / np.max(cc_a)).astype('uint8'), size)

        return np.dstack((cc_r, cc_g, cc_b, cc_a))

    def __read_object_mask(self, exr, size):
        """ read the segmentation of the objects from the exr file
        """

        index = np.fromstring(exr.channel('RenderLayer.IndexOB.X', FLOAT), dtype=np.float32)
        index = np.reshape(index, size)

        return index

    def __get_bounding_box(self, index_image, index):
        """ get the object bounding box in 2D
        :param the index / segmentation image
        :param the specified object index
        return the bounding box of the specified object and whether it is visible
        """
        H, W = index_image.shape
        rows = np.linspace(0, H-1, H)
        cols = np.linspace(0, W-1, W)
        us, vs = np.meshgrid(cols, rows)

        obj_mask = (index_image == index)
        u_mask = us[obj_mask]
        v_mask = vs[obj_mask]

        if u_mask.size == 0 or v_mask.size == 0 or obj_mask.sum() < 100:
            # if the object are not visible or too small to distinguish
            box = [0,0,0,0]
            valid = 0 # do not use this info
        else:
            lu, lv = int(u_mask.min()), int(v_mask.min())
            ru, rv = int(u_mask.max()), int(v_mask.max())
            box = [lu, lv, ru, rv]
            valid = 1

        return box, valid

    def __visualize(self, color, obj_boxes2D):
        """ Visualize the ground truth of
        """
        fig, ax = plt.subplots(1)
        ax.imshow(color)
        import matplotlib.patches as patches
        for k, v in obj_boxes2D.items():
            rect = patches.Rectangle((v[1], v[0]), v[3]-v[1], v[2]-v[0],
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate 3D shapes')
    parser.add_argument('--shape_id', type=str, default='None',
        help='the shapenet id')
    parser.add_argument('--seq_num', type=int, default=1,
        help='the number of sequences being generated')
    parser.add_argument('--start_index', type=int, default=0)

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    for idx in range(args.seq_num):
        post_fix = "{:06d}".format(idx + args.start_index)
        shapenet_parser = ShapeNetSceneParser(post_fix)
        shapenet_parser.run()
