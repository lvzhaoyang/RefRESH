import sys, os
import config
import os.path as osp
import pickle

from parse_static_scene import StaticSceneParser

BLENDER_PATH='~/develop/blender-2.79b'
dataset = 'bundlefusion'

scenes = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
keyframes = [1, 2, 5, 10, 20]

def total_file_num(params, stride):
    raw_path_pickle = osp.join(params['input_path'], scene+'.pkl')
    with open(raw_path_pickle, 'rb') as f:
        files = pickle.load(f, encoding='bytes')
        bg_poses = files['poses']

    total_num = len(bg_poses)
    cam_poses = []
    for idx in range(0, total_num, stride):
        pose = bg_poses[idx]
        if pose.size < 16:
            continue
        cam_poses.append(pose)

    return len(cam_poses)

def loop_file_existence(files):
    for f in files:
        if not osp.exists(f): 
            print('Missing {:}'.format(f))            
            return False

def check_exr_tmp_is_rendered(scene, stride):
    """ Return True if the intermediate openexr file is rendered. 
        Then we can skip the rendering process.        
    """

    params = config.load_file('configs/main_config', 'STATIC_3D_SCENE')
    total_num = total_file_num(params, stride)

    rendered_path = osp.join(params['tmp_path'], scene, 'keyframe_'+str(stride))
    for idx in range(total_num):
        candidate = osp.join(rendered_path, 'Image{:04d}.exr'.format(idx))
        if not osp.exists(candidate):    
            print('Missing {:}'.format(candidate))
            print('Need to re-render the scene {:}, keyframe {:}'.format(scene, stride))
            return False

    print('All openexrs have been correctly generated for scene {:}, keyframe {:}'.format(scene, stride))
    print('Move to the next')
    
    return True

def check_final_is_rendered(scene, stride):
    """ Return True if the final rendered files are complete. 
        Then we can skip the file generation process 
    """
    params = config.load_file('configs/main_config', 'STATIC_3D_SCENE')
    total_num = total_file_num(params, stride)

    output_pickle = osp.join(params['output_path'], scene, 'keyframe_'+str(stride), 'info.pkl')
    if not osp.exists(output_pickle):
        return False

    with open(output_pickle, 'rb') as f:
        files = pickle.load(f)

        if not (loop_file_existence(files['flow_forward']) or \
            loop_file_existence(files['flow_backward']) or \
            loop_file_existence(files['depth'])):
            print('Need to re-generates files for the scene {:}, keyframe {:}'.format(scene, stride))

    print('All files have been correctly generated for scene {:}, keyframe {:}'.format(scene, stride))

    if osp.exists(params['tmp_path']):
        print('Remove all temporary files {:}'.format(params['tmp_path']))
        os.system('rm -rf {:}'.format(params['tmp_path']))

    return True

def render_worker(scene, keyframe):

    tmp_exist = check_exr_tmp_is_rendered(scene, keyframe)
    final_exist = check_final_is_rendered(scene, keyframe)

    if not final_exist:
        if not tmp_exist:
            args = '-- --dataset {:} --scene {:} --stride {:}'.format(dataset, scene, keyframe)
            command = '{:}/blender --background --python \
                render_static_scenes.py {:}'.format(BLENDER_PATH, args)
            os.system(command)

        static_scene_parser = StaticSceneParser(dataset, scene, keyframe)
        static_scene_parser.run()

if __name__ == '__main__':
    
    from multiprocessing import Pool

    number_processes = 4

    # render all the scenes
    render_list = []
    for scene in scenes: 
        for keyframe in keyframes:
            render_list.append([scene, keyframe])

    p = multiprocessing.Pool(number_processes)
    p.map(render_worker, render_list)



