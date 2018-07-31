
import sys, os, time, argparse, random
import numpy as np
import io_utils

from pickle import load, dump
from os.path import join, dirname, realpath, exists
from scipy.misc import imread

import OpenEXR, Imath
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

class StaticSceneParser:

    def __init__(self, dataset_name = None, scene_name = None, stride = None,
        compare=False):

        self.params = io_utils.load_file('configs/main_config', 'STATIC_3D_SCENE')
        self.width = self.params['width']
        self.height= self.params['height']

        self.compare = compare

        if scene_name is None:
            scene_name = self.params['scene']
        if stride is None:
            stride = self.params['stride']

        print('parse the static scene {:s} stride {:d}'.format(scene_name, stride))

        scene_path_pickle = join(self.params['input_path'], scene_name+'.pkl')
        with open(scene_path_pickle, 'rb') as f:
            files = load(f)
            bg_color_files = files['color']
            bg_depth_files = files['depth']
            bg_poses = files['poses']
            bg_name = files['name']
            self.bg_calib = files['calib'] # calibration files

        self.total_num = len(bg_poses)
        self.cam_poses = []
        self.raw_images = []
        self.raw_depths = []
        # filter out all bad poses and mark them out
        for idx in range(0, self.total_num, stride):
            pose = bg_poses[idx]
            if pose.size < 16:
                continue
            self.cam_poses.append(pose)
            self.raw_images.append(bg_color_files[idx])
            self.raw_depths.append(bg_depth_files[idx])

        self.total_num = len(self.cam_poses)

        folder_name = join(bg_name, 'keyframe_' + str(stride))

        output_path = self.params['output_path']
        output_path = join(output_path, folder_name)
        self.output_path = output_path

        tmp_path = self.params['tmp_path']
        tmp_path = join(tmp_path, folder_name)
        self.tmp_path = tmp_path

    def run(self):

        print('generate output for {:s}'.format(self.output_path))

        rendered_dir        = join(self.output_path, 'rendered')
        depth_dir           = join(self.output_path, 'depth')
        flow_forward_dir    = join(self.output_path, 'flow_forward')
        flow_backward_dir   = join(self.output_path, 'flow_backward')
        flowviz_forward_dir = join(self.output_path, 'flow_vis_forward')
        flowviz_backward_dir= join(self.output_path, 'flow_vis_backward')
        invalid_dir         = join(self.output_path, 'invalid')

        info = {'raw_color':        [], 
                'raw_depth':        [], 
                'rendered':         [],
                'depth':            [],
                'flow_forward':     [],
                'flow_backward':    [],
                'flowviz_forward':  [],
                'flowviz_backward': [],
                'pose':             [],
                'invalid':          [],
                'calib':            self.bg_calib}

        if self.compare:
            color_compare_dir = join(self.output_path, 'compare_color')
            depth_compare_dir = join(self.output_path, 'compare_depth')
            io_utils.create_directory(color_compare_dir)
            io_utils.create_directory(depth_compare_dir)

        for idx in range(0, self.total_num):

            exr_file = join(self.tmp_path, 'Image{:04d}.exr'.format(idx))
            exr= OpenEXR.InputFile(exr_file)
            size = (self.height, self.width)

            invalid_mask = np.zeros(size, np.uint8)

            # process flow
            forward_flow, backward_flow = self.__read_flow(exr, size)
            flow_forward_vis = io_utils.flow_visualize(forward_flow)
            flow_backward_vis= io_utils.flow_visualize(backward_flow)

            # process depth
            depth, invalid_depth = self.__read_depth(exr, size)
            invalid_mask[invalid_depth] = 255                    

            # process rendered color image
            color = self.__read_color(exr, size)

            output_name = str(idx).zfill(6)
            print('generate file: {:}'.format(output_name))
            filename_flo = output_name+'.flo'
            filename_png = output_name+'.png'
            flow_forward_file       = join(flow_forward_dir, filename_flo) 
            flow_backward_file      = join(flow_backward_dir,filename_flo) 
            flowviz_forward_file    = join(flowviz_forward_dir, filename_png)
            flowviz_backward_file   = join(flowviz_backward_dir,filename_png)
            depth_file              = join(depth_dir,   filename_png)
            invalid_mask_file       = join(invalid_dir, filename_png)
            rendered_color_file     = join(rendered_dir,filename_png)

            io_utils.flow_write(flow_forward_file,  forward_flow)
            io_utils.flow_write(flow_backward_file, backward_flow)
            io_utils.image_write(flowviz_forward_file, flow_forward_vis)
            io_utils.image_write(flowviz_backward_file,flow_backward_vis)    
            io_utils.pngdepth_write(depth_file, depth)
            io_utils.image_write(invalid_mask_file, invalid_mask)
            io_utils.image_write(rendered_color_file, color)

            info['rendered'].append(rendered_color_file)
            info['flow_forward'].append(flow_forward_file)
            info['flow_backward'].append(flow_backward_file)
            info['flowviz_forward'].append(flowviz_forward_file)
            info['flowviz_backward'].append(flowviz_forward_file)
            info['depth'].append(depth_file)
            info['invalid'].append(invalid_mask_file)
            info['pose'].append(self.cam_poses[idx])
            info['raw_color'].append('../'+ 
                self.raw_images[idx][self.raw_images[idx].find('data/RefRESH'):])
            info['raw_depth'].append('../'+
                self.raw_depths[idx][self.raw_depths[idx].find('data/RefRESH'):])

            # save the output into a video with all sources
            if self.compare:
                raw_color = imread(self.raw_images[idx])
                raw_depth = imread(self.raw_depths[idx])
                rendered_color = imread(rendered_raw_file)

                color_image_compare = np.zeros((self.height, self.width*2, 3), np.uint8)
                depth_image_compare = np.zeros((self.height, self.width*2), np.uint16)

                color_image_compare[:, :self.width, :] = raw_color
                color_image_compare[:, self.width:, :] = rendered_color[:, :, :3]
                depth_image_compare[:, :self.width] = raw_depth
                depth_image_compare[:, self.width:] = depth*1e3

                io_utils.image_write(
                    join(color_compare_dir, output_name+'.png'), 
                    color_image_compare)
                io_utils.depth_write(
                    join(depth_compare_dir, output_name+'.png'), 
                    depth_image_compare)

        # write all the final files into a pickle file
        dataset_path = join(self.output_path, 'info.pkl')
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
        invalid_depth = depth > 1e2
        depth[invalid_depth] = 0 # set the depth in invalid region to be 0

        return depth, invalid_depth

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

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic outputs')
    parser.add_argument('--compare', type=bool, default=False, help='Generate pairwise comparison outputs of rendered vs raw')
    parser.add_argument('--dataset', type=str, default='None', help='the dataset name')
    parser.add_argument('--scene', type=str, default='None', help='the scene name in the dataset')
    parser.add_argument('--stride', type=int, default = 0, help='the keyframes set for background rendering')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    dataset, scene, stride = None, None, None
    if args.dataset != 'None':
        dataset = args.dataset
    if args.scene != 'None':
        scene = args.scene
    if args.stride != 0:
        stride = args.stride

    bp = StaticSceneParser(dataset, scene, stride, args.compare)

    bp.run()
