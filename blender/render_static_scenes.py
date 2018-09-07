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

import sys, os, random, math, time
import numpy as np

from os import getenv, remove
from os.path import join, dirname, realpath, exists
from glob import glob
from pickle import load
from random import choice

import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam
from mathutils import Matrix, Vector, Quaternion, Euler

sys.path.insert(0, ".")
from io_utils import load_file
from utils import world_to_blender, set_intrinsic

class StaticSceneRender:
    """ Render the 3D static scene given the camera trajectory 
    """

    start_time = None

    def __init__(self, dataset_name = None, scene_name = None, stride = None):

        self.start_time = time.time()

        self.log_message("Importing the 3D static scene info.")

        self.params = load_file('configs/main_config', 'STATIC_3D_SCENE')

        if dataset_name is None:
            dataset_name = self.params['dataset']
        if scene_name is None:
            scene_name = self.params['scene']
        if stride is None:
            stride = self.params['stride']

        data_folder = self.params['data_folder']
        scene_path_pickle = join(self.params['input_path'], scene_name+'.pkl')
        with open(scene_path_pickle, 'rb') as f:
            files = load(f, encoding='bytes')
            bg_color_files = files['color']
            bg_depth_files = files['depth']
            bg_poses = files['poses']
            bg_mesh_file = files['mesh']
            bg_name = files['name']
            bg_calib = files['calib']

        self.total_num = len(bg_poses)
        self.cam_poses = []
        # filter out all bad poses and mark them out
        for idx in range(0, self.total_num, stride):
            pose = bg_poses[idx]
            if pose.size < 16:
                continue
            self.cam_poses.append(pose)

        self.total_num = len(self.cam_poses)
        print('There will be {:d} poses rendered for the \
            3D scene {:}, keyframe {:}'.format(self.total_num, scene_name, stride))
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = self.total_num

        self.color_K = bg_calib['color_intrinsic']
        self.depth_K = bg_calib['depth_intrinsic']

        self.init_render_settings()

        # where all the openexr files are written to
        folder_name = join(bg_name, 'keyframe_' + str(stride))
        self.tmp_path = join(self.params['tmp_path'], folder_name)

        self.init_scene(join(data_folder, bg_mesh_file))

    def run_rendering(self):
        """ Run the rendering script.
        """
        self.log_message('Render the static scene with the given trajectory...')

        get_real_frame = lambda ifr: ifr
        bpy_scene = bpy.context.scene
        ''' ---------------------- LOOP TO RENDER  ------------------------- '''
        for frame_idx in range(0, self.total_num):
            bpy_scene.frame_set(get_real_frame(frame_idx))

            bpy_scene.render.filepath = join(self.tmp_path, 'Image%04d.exr' % get_real_frame(frame_idx))

            self.log_message("Rendering frame %d" % frame_idx)

            # Render
            bpy.ops.render.render(write_still=True)

    def init_render_settings(self):

        self.log_message("Setup Blender Cycles Render Engine")

        bpy_scene = bpy.context.scene
        bpy_render = bpy_scene.render

        bpy_scene.cycles.shading_system = True
        bpy_scene.use_nodes = True
        
        bpy_render.use_overwrite = False
        bpy_render.use_placeholder = True
        bpy_render.use_antialiasing = False

        bpy_render.layers["RenderLayer"].use_pass_vector = self.params['output_types']['gtflow']
        bpy_render.layers["RenderLayer"].use_pass_normal = self.params['output_types']['normal']        
        bpy_render.layers["RenderLayer"].use_pass_z = self.params['output_types']['depth']
        bpy_render.layers['RenderLayer'].use_pass_emit   = False

        # set render size
        bpy_render.resolution_x = self.params['width']
        bpy_render.resolution_y = self.params['height']

        bpy_render.resolution_percentage = 100
        # bpy_render.image_settings.file_format = 'PNG'
        bpy_render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        bpy_render.image_settings.color_mode = 'RGBA'

        # settings to speed up the blender rendering speed
        # Since we don't care too much about the rendered images, these setttings are
        # pretty redundant
        # for tile rendering
        bpy_render.use_raytrace = False
        bpy_render.tile_x = 512
        bpy_render.tile_y = 512
        bpy_scene.cycles.max_bounces = 1
        bpy_scene.cycles.samples = 64

        # choose the rendering engine
        # cycles_preferences = bpy.context.user_preferences.addons['cycles'].preferences
        # cycles_preferences.compute_device_type = "CUDA"

        # # Specify the gpu for rendering
        # for device in cycles_preferences.devices:
        #     device.use = False
        # cycles_preferences.devices[7].use = True

    def init_scene(self, mesh_path):

        self.log_message('Initialize the scene')

        # delete the default cube (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete(use_global=False)

        self.setup_camera_sequence()

        self.log_message("Importing static 3D scene meshes. This might takes some time...")
        bpy.ops.import_mesh.ply(filepath=mesh_path)
        obj = bpy.context.active_object
        # render vertex color
        # Get material
        mat = bpy.data.materials.new(name="Material_static_scene")
        mat.use_vertex_color_paint = True
        # Assign it to object
        if obj.data.materials:
            # assign to 1st material slot
            obj.data.materials[0] = mat
        else:
            # no slots
            obj.data.materials.append(mat)

    def setup_camera_sequence(self):
        """ Set up the blender camera intrinsics and extrinsics according to the ground truth camera
        """
        lamp = bpy.data.objects['Lamp']

        # set camera properties and initial position
        bpy.ops.object.select_all(action='DESELECT')
        bpy_camera_obj = bpy.data.objects['Camera']
        bpy_scene = bpy.context.scene
        bpy_scene.objects.active = bpy_camera_obj

        # set camera intrisic
        K = [self.color_K[0][0], self.color_K[1][1], self.color_K[0,2], self.color_K[1,2]]

        set_intrinsic(K, bpy_camera_obj, bpy_scene, self.params['height'], self.params['width'])

        # set the camera trajectory according to the ground truth trajectory
        bpy_scene = bpy.context.scene
        for frame_idx in range(0, self.total_num):

            bpy_scene.frame_set(frame_idx)

            bpy_camera_obj.matrix_world = world_to_blender(Matrix(self.cam_poses[frame_idx]))
            bpy_camera_obj.keyframe_insert('location', frame=frame_idx)
            bpy_camera_obj.keyframe_insert('rotation_euler', frame=frame_idx)

            # move the lamp together with the light
            lamp.matrix_world = bpy_camera_obj.matrix_world
            lamp.keyframe_insert('location', frame=frame_idx)
            lamp.keyframe_insert('rotation_euler', frame=frame_idx)

            bpy_scene.update()

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic background')
    # parser.add_argument('--idx', type=int, default=0, help='default idx')
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

    rs = StaticSceneRender(dataset, scene, stride)
    rs.run_rendering()
