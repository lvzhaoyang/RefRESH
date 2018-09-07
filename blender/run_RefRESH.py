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

import sys, os
import pickle
import random

import os.path as osp
import numpy as np
import io_utils

def total_file_num(params, scene, keyframe):
    raw_path_pickle = osp.join(params['input_path'], scene, 
        'keyframe_{:}'.format(keyframe), 'info.pkl')
    with open(raw_path_pickle, 'rb') as f:
        files = pickle.load(f)
        bg_poses = files['pose']

    total_num = len(bg_poses)
    cam_poses = []
    for idx in range(0, total_num, keyframe):
        pose = bg_poses[idx]
        cam_poses.append(pose)

    return len(cam_poses)

def loop_file_existence(files):
    for f in files:
        if not osp.exists(f): 
            print('Missing {:}'.format(f))            
            return False
        return True

def check_exr_tmp_is_rendered(exr_path, start, end):
    """ Return True if the intermediate openexr file is rendered. 
        Then we can skip the rendering process.        
    """
    for idx in range(start, end):
        candidate = osp.join(exr_path, 'Image{:04d}.exr'.format(idx))
        if not osp.exists(candidate):    
            print('Missing {:}'.format(candidate))
            print('Need to re-render {:}'.format(exr_path))
            return False

    print('All openexrs have been correctly generated for {:}'.format(exr_path))
    print('Move to the next')
    
    return True

def check_background_is_rendered(scene, stride):
    """ Return True if the background rendered files are complete. 
    """
    params = io_utils.load_file('configs/main_config', 'SYNTH_HUMAN')
    total_num = total_file_num(params, scene, stride)

    output_pickle = osp.join(params['input_path'], scene, 'keyframe_{:}'.format(stride), 'info.pkl')
    if not osp.exists(output_pickle):
        print('Background files missing for the scene {:}, keyframe {:}'.format(scene, stride))
        print('Skip the foreground generation.')
        return False

    with open(output_pickle, 'rb') as f:
        files = pickle.load(f)

        if not (loop_file_existence(files['flow_forward']) or \
            loop_file_existence(files['flow_backward']) or \
            loop_file_existence(files['depth'])):
            print('Background files missing for the scene {:}, keyframe {:}'.format(scene, stride))
            print('Skip the foreground generation.')
            return False

    print('All background files have been correctly generated for scene {:}, keyframe {:}'.format(scene, stride))
    return True

def check_final_is_rendered(render_path, tmp_path, start, end):
    """ Return True if the final rendered files are complete. 
        Then we can skip the file generation process 
    """
    output_pickle = osp.join(render_path, 'info.pkl')
    if not osp.exists(output_pickle):
        return False

    with open(output_pickle, 'rb') as f:
        files = pickle.load(f)

        if not (loop_file_existence(files['flow_forward']) or \
            loop_file_existence(files['flow_backward']) or \
            loop_file_existence(files['depth'])):
            print('Need to re-generates files for the {:}'.format(render_path))

    print('All files have been correctly generated for scene {}'.format(render_path))

    if osp.exists(tmp_path):
        print('Remove all temporary files {:}'.format(tmp_path))
        os.system('rm -rf {:}'.format(tmp_path))

    return True

def render_worker(settings):

    scene, keyframe = settings

    # do not render the scenes that without background
    if not check_background_is_rendered(scene, keyframe): return 

    params = io_utils.load_file('configs/main_config', 'SYNTH_HUMAN')
    fg_diversity_avg    = params['fg_diversity_avg']
    fg_stride_avg       = params['fg_stride_avg']
    fg_frame_number     = params['fg_frame_number']

    total_frame_num = total_file_num(params, scene, keyframe)

    start, end = None, None
    for start in range(0, total_frame_num, fg_frame_number):
        end = min(start + fg_frame_number, total_frame_num)

        tmp_path = osp.join(params['tmp_path'], scene, 'keyframe_{:}'.format(keyframe), 
            '{:04d}_{:04d}'.format(start, end))
        render_path = osp.join(params['output_path'], scene, 'keyframe_{:}'.format(keyframe), 
            '{:04d}_{:04d}'.format(start, end))

        if check_final_is_rendered(render_path, tmp_path, start, end):
            continue

        if not check_exr_tmp_is_rendered(tmp_path, start, end):
            #  the number of humans are all randomly generated
            random.seed()
            fg_diversity = int(np.random.normal(fg_diversity_avg, 1, 1)[0])
            fg_stride = int(np.random.normal(fg_stride_avg, 10, 1)[0])

            # run each trajectory at a time
            args = '--fg_density {:d} --fg_stride {:d} --bg_scene {:s} --bg_stride {:d} --bg_start {:d} --bg_end {:d}'.format(fg_diversity, fg_stride, scene, keyframe, start, end)
            command = '{:}/blender --background --python render_humans_from_trajectories.py -- {:}'.format(BLENDER_PATH, args)
            os.system(command)

        from parse_humans_from_trajectories import HumanSceneParser
        dynamic_scene_parser = HumanSceneParser(scene, keyframe, start, end)
        dynamic_scene_parser.run()

        print('Remove all temporary files {:}'.format(tmp_path))
        os.system('rm -rf {:}'.format(tmp_path))

if __name__ == '__main__':

    BLENDER_PATH='~/develop/blender-2.79b'

    import argparse
    parser = argparse.ArgumentParser(description='Run BundleFusion')
    parser.add_argument('--index', type=int, default = -1, 
        help='set the index to run the jobs. The default is set to -1 and run all the jobs.')
    parser.add_argument('--processes', type=int, default=1, 
        help='the number of processes to run multi-jobs if execute run_all')
    args = parser.parse_args()

    scenes = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
    # Note that 10, 20 keyframes are too large for training the scene flow
    # You can choose not to render them
    keyframes = [1, 2, 5, 10, 20] 

    render_list = []
    for scene in scenes:
        for keyframe in keyframes:
            render_list.append([scene, keyframe])

    if args.index < 0: 
        print('Run all the jobs')
        import multiprocessing
        # render all the scenes
        p = multiprocessing.Pool(args.processes)
        p.map(render_worker, tuple(render_list))
    else: 
        render_worker(tuple(render_list[args.index]))        
