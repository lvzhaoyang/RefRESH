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
import numpy as np
import io_utils

from pickle import load, dump
from os.path import join, dirname, realpath, exists
from scipy.misc import imread

import OpenEXR, Imath
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

class ShapeNetSceneParser:

    def __init__(self, post_fix = ''):

        self.params = io_utils.load_file('config/shapenet_config', 'SHAPENET_MODELS')

        self.output_path = osp.join(self.params['output_path'], post_fix)
        self.tmp_path = osp.join(self.params['tmp_path'], post_fix)

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

        for idx in range(0, self.params['trajectory']['views']):

            exr_file = osp.join(self.tmp_path, 'Image{:04d}.exr'.format(idx))
            exr = OpenEXR.InputFile(exr_file)
            size = (self.params['height'], self.params['width'])

            invalid_mask = np.zeros(size, np.uint8)

            forward_flow, backward_flow = self.__read_flow(exr, size)
            flow_forward_vis = io_utils.flow_visualize(forward_flow)
            flow_backward_vis= io_utils.flow_visualize(backward_flow)

            # process depth
            depth, invalid_depth = self.__read_depth(exr, size)
            invalid_mask[invalid_depth] = 255

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
