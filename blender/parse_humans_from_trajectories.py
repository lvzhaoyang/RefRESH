import sys, os, time, random, pickle

import numpy as np
import matplotlib.pyplot as plt
import io_utils

from os.path import join, exists, isdir
from scipy.misc import imread

import OpenEXR, array, Imath
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def create_directory(target_dir):
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

class HumanSceneParser:

    start_time = None

    def __init__(self, bg_scene, bg_stride, bg_start, bg_end):

        self.start_time = time.time()

        params = io_utils.load_file('configs/main_config', 'SYNTH_HUMAN')

        if bg_scene is None:
            bg_scene = params['bg_scene']
        if bg_stride is None:
            bg_stride = params['bg_stride']
        if bg_start is None:
            bg_start = params['bg_start']
        if bg_end is None:
            bg_end = params['bg_end']

        self.bg_start, self.bg_end = bg_start, bg_end

        #######################################################################
        self.log_message('Parsing information for scene {:s} keyframe {:d} start at{:d} end at {:d}'.format(bg_scene, bg_stride, bg_start, bg_end))

        self.width = params['width']
        self.height= params['height']

        #######################################################################
        self.log_message('Set up background information')
        bg_base_path = params['bg_base_path']

        background_path = join(bg_base_path, bg_scene, 'keyframe_{:}'.format(bg_stride), 'info.pkl')
        with open(background_path, 'rb') as f:
            files = pickle.load(f)
            self.bg_color_files         = files['raw_color']
            self.bg_raw_depth_files     = files['raw_depth']
            self.bg_render_files        = files['rendered']
            self.bg_depth_files         = files['depth']
            self.bg_flow_forward_files  = files['flow_forward']
            self.bg_flow_backward_files = files['flow_backward']
            self.bg_poses               = files['pose']
            self.bg_calib               = files['calib']
            self.bg_invalid             = files['invalid']

        self.total_num = bg_end - bg_start

        #######################################################################
        self.log_message('Set up output folder')
        output_path = params['output_path']
        out_folder_name = join(bg_scene, 'keyframe_{:}'.format(bg_stride), 
            '{:04d}_{:04d}'.format(bg_start, bg_end))
        self.output_path = join(output_path, out_folder_name)
        print('The final output will be written to {:s}'.format(self.output_path))

        output_types = params['output_types']
        tmp_folder_name = join(bg_scene, 'keyframe_{:}'.format(bg_stride), 
            '{:04d}_{:04d}'.format(bg_start, bg_end))
        self.tmp_path = join(params['tmp_path'], tmp_folder_name)
        output_types['rgb_video.mp4'] = True

    def run(self):

        print('generate output for {:s}'.format(self.output_path))

        W, H = self.width, self.height

        depth_dir           = join(self.output_path, 'depth')
        depth_raw_dir       = join(self.output_path, 'depth_raw')

        flow_forward_dir    = join(self.output_path, 'flow_forward')
        flow_backward_dir   = join(self.output_path, 'flow_backward')
        flowviz_forward_dir = join(self.output_path, 'flow_vis_forward')
        flowviz_backward_dir= join(self.output_path, 'flow_vis_backward')

        rigidity_dir        = join(self.output_path, 'rigidity')
        color_dir           = join(self.output_path, 'color')
        invalid_dir         = join(self.output_path, 'invalid')

        info = {'color':            [], # composited image with raw image as background
                'raw_depth':        [], # composited depth with raw depth as background
                'depth':            [], # the composited depth
                'depth_raw':        [], # the composited depth using the raw background
                'flow_forward':     [], 
                'flow_backward':    [],   
                'flowviz_forward':  [], 
                'flowviz_backward': [], 
                'pose':             [], 
                'rigidity':         [], # rigidity mask for background
                'invalid':          [], # the composited invalid mask
                'calib':            self.bg_calib}  # the camera intrinsic information

        # overlap determined by stride (# subsampled frames to skip)
        # LOOP OVER FRAMES
        for idx in range(0, self.total_num):
            # since we only have the backward flow, we will ignore the first frame
            self.log_message("Processing frame %d" % idx)
            output_name = str(idx).zfill(6)

            bg_idx = self.bg_start + idx

            exr_file = join(self.tmp_path, 'Image{:04d}.exr'.format(idx))
            exr = OpenEXR.InputFile(exr_file)
            size = (self.height, self.width)

            # process depth
            fg_depth, rigidity = self.__read_depth(exr, size)
            invalid_mask = imread(self.bg_invalid[bg_idx]) # background invalid
            depth = io_utils.pngdepth_read(self.bg_depth_files[bg_idx])
            depth[rigidity] = fg_depth[rigidity]

            # process color 
            fg_color = self.__read_color(exr, size)
            color = imread(self.bg_color_files[bg_idx], mode='RGBA') #background color
            color[rigidity] = fg_color[rigidity]

            # process flow             
            fg_forward_flow, fg_backward_flow = self.__read_flow(exr, size)
            forward_flow = np.dstack(io_utils.flow_read_from_flo(self.bg_flow_forward_files[bg_idx]))
            backward_flow= np.dstack(io_utils.flow_read_from_flo(self.bg_flow_backward_files[bg_idx]))
            forward_flow[rigidity] = fg_forward_flow[rigidity]
            backward_flow[rigidity]= fg_backward_flow[rigidity]

            flow_forward_vis = io_utils.flow_visualize(forward_flow)
            flow_backward_vis= io_utils.flow_visualize(backward_flow)            

            # process rendered color image
            filename_png = output_name + '.png'
            filename_flo = output_name + '.flo'
            color_file              = join(color_dir, filename_png)
            depth_file              = join(depth_dir, filename_png)
            rigidity_file           = join(rigidity_dir, filename_png)
            flow_forward_file       = join(flow_forward_dir, filename_flo)
            flow_backward_file      = join(flow_backward_dir, filename_flo)
            flowviz_forward_file    = join(flowviz_forward_dir, filename_png)
            flowviz_backward_file   = join(flowviz_backward_dir,filename_png)
            
            io_utils.image_write(color_file, color)
            io_utils.pngdepth_write(depth_file, depth)
            io_utils.flow_write(flow_forward_file, forward_flow)
            io_utils.flow_write(flow_backward_file, backward_flow)
            io_utils.image_write(rigidity_file, rigidity*255)
            io_utils.image_write(flowviz_forward_file, flow_forward_vis)
            io_utils.image_write(flowviz_backward_file,flow_backward_vis)    

            info['color'].append(color_file)
            info['rigidity'].append(rigidity_file)
            info['depth'].append(depth_file)
            info['flow_forward'].append(flow_forward_file)
            info['flow_backward'].append(flow_backward_file)
            info['flowviz_forward'].append(flowviz_forward_file)
            info['flowviz_backward'].append(flowviz_forward_file)
            info['invalid'].append(self.bg_invalid[bg_idx])
            info['pose'].append(self.bg_poses[bg_idx])

        # write all the final files into a pickle file
        dataset_path = join(self.output_path, 'info.pkl')
        with open(dataset_path, 'wb') as output:
            pickle.dump(info, output)

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

    def clean_up(self):
        # cleaning up tmp
        if self.tmp_path != "" and self.tmp_path != "/":
            self.log_message("Cleaning up tmp")
            os.system('rm -rf %s' % self.tmp_path)

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
        rigidity = depth < 1e3
        depth[~rigidity] = 0

        return depth, rigidity

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
    parser = argparse.ArgumentParser(description='Generate synthetic dataset images.')
    parser.add_argument('--fg_density', type=int, default=0, help='Every N frames we will load an object into the scene.')
    parser.add_argument('--fg_stride', type=int, default=0, help='The number of stride when we load')
    parser.add_argument('--bg_scene', type=str, default='None', help='The scene trajectory to be used as background')
    parser.add_argument('--bg_stride', type=int, default=0, help='The number of background stride')
    parser.add_argument('--bg_start', type=int, default=-1, help='The start frame in the background trajectory')
    parser.add_argument('--bg_end', type=int, default=-1, help='The end frame in the background trajectory')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    bg_scene, bg_stride = None, None
    if args.bg_scene != 'None':
        bg_scene = args.bg_scene
    if args.bg_stride != 0:
        bg_stride = args.bg_stride
    if args.bg_start != -1:
        bg_start = args.bg_start
    if args.bg_end != -1:
        bg_end = args.bg_end

    scene_parser = HumanSceneParser(bg_scene, bg_stride, bg_start, bg_end)

    scene_parser.run()
