import sys, os
import io_utils
import os.path as osp
import pickle

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

    params = io_utils.load_file('configs/main_config', 'STATIC_3D_SCENE')
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
    params = io_utils.load_file('configs/main_config', 'STATIC_3D_SCENE')
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

def render_worker(settings):

    scene, keyframe = settings

    final_exist = check_final_is_rendered(scene, keyframe)

    if not final_exist:

        tmp_exist = check_exr_tmp_is_rendered(scene, keyframe)
        if not tmp_exist:
            args = '-- --dataset {:} --scene {:} --stride {:}'.format(dataset, scene, keyframe)
            command = '{:}/blender --background --python \
                render_static_scenes.py {:}'.format(BLENDER_PATH, args)
            os.system(command)

        from parse_static_scene import StaticSceneParser
        static_scene_parser = StaticSceneParser(dataset, scene, keyframe)
        static_scene_parser.run()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Run BundleFusion')
    parser.add_argument('--index', type=int, default = -1, 
        help='set the index to run the jobs. The default is set to -1 and run all the jobs.')
    parser.add_argument('--processes', type=int, default=1, 
        help='the number of processes to run multi-jobs if execute run_all')
    args = parser.parse_args()

    BLENDER_PATH='~/develop/blender-2.79b'
    dataset = 'bundlefusion'

    scenes = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
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



