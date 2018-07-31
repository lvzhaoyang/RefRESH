#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Pre-process the BundleFusion 
"""

import os, pickle, glob
import numpy as np
from os.path import join

def parse_calibration_file(file):

    calib = {
        'color_width': -1,
        'color_height': -1,
        'depth_width': -1,
        'depth_height': -1,
        'depth_shift': -1,
        'color_intrinsic': -1,
        'color_extrinsic': -1,
        'depth_intrinsic': -1,
        'depth_extrinsic': -1,
        'frame_size': -1
    }

    calib = {}

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            info = line.split('=')

            if 'colorWidth' in info[0]:
                calib['color_width'] = int(info[1].rstrip())
            elif 'colorHeight' in info[0]:
                calib['color_height'] = int(info[1].rstrip())
            elif 'depthWidth' in info[0]:
                calib['depth_width'] = int(info[1].rstrip())
            elif 'depthHeight' in info[0]:
                calib['depth_height'] = int(info[1].rstrip())
            elif 'depthShift' in info[0]:
                calib['depth_shift'] = int(info[1].rstrip())
            elif 'ColorIntrinsic' in info[0]:
                intrinsic = np.fromstring(info[1].rstrip(), dtype=float, sep = ' ')
                calib['color_intrinsic'] = np.reshape(intrinsic, (4, 4))
            elif 'DepthIntrinsic' in info[0]:
                intrinsic = np.fromstring(info[1].rstrip(), dtype=float, sep = ' ')
                calib['depth_intrinsic'] = np.reshape(intrinsic, (4, 4))
            elif 'frames.size' in info[0]:
                calib['frame_size'] = int(info[1].rstrip())
    return calib

def write_files_into_pickle(local_dir, scene):

    calib_file = join(local_dir, scene, 'info.txt')

    calib = parse_calibration_file(calib_file)
    color_files, depth_files, poses = [], [], []
    for idx in range(calib['frame_size']):
        frame_prefix = 'frame-' + str(idx).zfill(6)

        color_file = join(local_dir, scene, frame_prefix + '.color.jpg')
        depth_file = join(local_dir, scene, frame_prefix + '.depth.png')
        pose_file  = join(local_dir, scene, frame_prefix + '.pose.txt')

        color_files.append(os.path.abspath(color_file))
        depth_files.append(os.path.abspath(depth_file))

        pose = np.loadtxt(pose_file)
        poses.append(pose)

    # the ply file
    mesh_file = join(local_dir, scene + '.ply')

    info = {'name': scene,
        'calib': calib,
        'color': color_files,
        'depth': depth_files,
        'poses': poses,
        'mesh': mesh_file}

    # save as python pickle (python 3)
    dataset_path = join(local_dir, scene + '.pkl')
    with open(dataset_path, 'wb') as output:
        pickle.dump(info, output)

if __name__ == '__main__':

    scenes = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']

    for scene in scenes:
        write_files_into_pickle('data/RefRESH/BundleFusion/raw', scene)
