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
import hashlib

from os import getenv, remove
from os.path import join, dirname, realpath, exists, isdir
from glob import glob
from pickle import load
from random import choice

import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.insert(0, ".")
from motion_surreal import *
from motion_mechanism import *
from io_utils import load_file
from utils import world_to_blender, set_intrinsic
from scipy.misc import imread

def restart_blender_scene():
    for obj in bpy.data.objects.values():
        obj.select=False
    bpy.context.scene.objects.active = None

class DynamicHumanRender:

    start_time = None

    frame_per_clip = 200 # We will generate 200 poses per clip

    def __init__(self, fg_density, fg_stride, bg_scene, bg_stride, bg_start, bg_end):

        self.start_time = time.time()

        self.params = load_file('configs/main_config', 'SYNTH_HUMAN')

        if bg_scene is None:
            bg_scene = self.params['bg_scene']
        if bg_stride is None:
            bg_stride = self.params['bg_stride']
        if bg_start is None:
            bg_start = self.params['bg_start']
        if bg_end is None:
            bg_end = self.params['bg_end']
        if fg_density is None:
            self.fg_total_number = self.params['fg_objects']
        else:
            self.fg_total_number = max(int((bg_end - bg_start) / fg_density), 1)
        if fg_stride is None:
            fg_stride = self.params['fg_stride']
            self.fg_stride = fg_stride
        else:
            self.fg_stride = fg_stride

        self.bg_start, self.bg_end = bg_start, bg_end
        self.scene_name, self.scene_stride = bg_scene, bg_stride

        base_config_path = self.params['data_folder']

        #####################################################################
        self.log_message("Setup Blender")
        scene = bpy.context.scene
        scene = bpy.data.scenes['Scene']
        scene.render.engine = 'CYCLES'
        scene.cycles.shading_system = True
        scene.use_nodes = True

        #####################################################################
        # import idx info with the format (name, split)
        self.log_message("Importing idx info pickle")
        idx_info = load(open(join(base_config_path, "pkl/idx_info.pickle"), 'rb'))
        # random load foreground indices
        idx_info_len = len(idx_info)
        fg_indices = [int(idx_info_len*random.random()) for i in range(idx_info_len)]

        self.fg_indices_info = []
        for idx in fg_indices:
            self.fg_indices_info.append(idx_info[idx])

        #######################################################################
        self.log_message("Loading the smpl data")
        smpl_data_folder = self.params['smpl_data_folder']
        smpl_data_filename = self.params['smpl_data_filename']
        self.smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

        ######################################################################
        self.log_message('Set up the camera information')
        self.init_camera(bg_scene, bg_stride, bg_start, bg_end)

        ########################################################################
        # the closing option grey, nongrey or all
        clothing_option = self.params['clothing_option']
        self.log_message('Set up foreground information. clothing: %s' % clothing_option)
        genders = {0: 'female', 1: 'male'}
        # pick several foreground objects with random gender and clothing
        self.clothing_names = []
        for idx in range(self.fg_total_number):
            gender = choice(genders)

            with open( join(smpl_data_folder, 'textures',
                '%s_%s.txt' % ( gender, self.fg_indices_info[idx]['use_split'] ) ) ) as f:
                txt_paths = f.read().splitlines()
            # if using only one source of clothing
            if clothing_option == 'nongrey':
                txt_paths = [k for k in txt_paths if 'nongrey' in k]
            elif clothing_option == 'grey':
                txt_paths = [k for k in txt_paths if 'nongrey' not in k]
            # random clothing texture
            cloth_img_name = choice(txt_paths)
            cloth_img_name = join(smpl_data_folder, cloth_img_name)
            self.clothing_names.append([gender, cloth_img_name])

        ######################################################################
        self.log_message('Prepare for output directory')
        self.init_directories(bg_scene, bg_start, bg_end, bg_stride)

        # >> don't use random generator before this point <<
        # initialize RNG with seeds from sequence id 
        s = "synth_data:{:d}, {:d}, {:s}, {:d}".format(fg_density, fg_stride, bg_scene, bg_stride)
        seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        self.log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
        random.seed(seed_number)
        np.random.seed(seed_number)

        output_types = self.params['output_types']
        if(output_types['vblur']):
            vblur_factor = np.random.normal(0.5, 0.5)
            self.params['vblur_factor'] = vblur_factor

        #######################################################################
        # grab clothing names
        self.log_message('Set up blender node')
        bg_img = bpy.data.images.load(self.bg_color_files[0])

        # self.create_composite_nodes(scene.node_tree, img=bg_img)

    def init_directories(self, bg_scene, bg_start, bg_end, bg_stride):

        folder_name = join(bg_scene, 'keyframe_{:}'.format(bg_stride), 
            '{:04d}_{:04d}'.format(bg_start, bg_end))

        self.tmp_path = join(self.params['tmp_path'], folder_name)
        if not isdir(self.tmp_path): os.makedirs(self.tmp_path)

        # create copy-spher.harm. directory if not exists
        sh_dir = join(self.tmp_path, 'sphere_harmonic_lighting')
        if not isdir(sh_dir): os.makedirs(sh_dir)
        self.sh_dst = join(sh_dir, 'sh_{:s}.osl'.format(bg_scene))
        os.system('cp sphere_harmonic_lighting/sh.osl {:s}'.format(self.sh_dst))

    def init_camera(self, bg_scene, bg_stride, bg_start, bg_end):
        """ Initialize the background and camera settings according to 
            the background scenes information.
        """
        # load the pickle file generated from background rendering
        bg_base_path = self.params['bg_base_path']
        background_path = join(bg_base_path, bg_scene, 
            'keyframe_{:}'.format(bg_stride), 'info.pkl')

        with open(background_path, 'rb') as f:
            files = load(f, encoding='bytes')
            bg_color_files = files['raw_color']
            bg_depth_files = files['raw_depth']
            bg_poses = files['pose']
            bg_calib = files['calib'] # calibration files

        bg_end = min(bg_end, len(bg_poses))

        self.bg_color_files = []
        self.bg_depth_files = []
        self.cam_poses = []
        for idx in range(bg_start, bg_end):
            pose = bg_poses[idx]
            if pose.size < 16:
                continue # prune bad poses
            self.cam_poses.append(pose)
            self.bg_color_files.append(bg_color_files[idx])
            self.bg_depth_files.append(bg_depth_files[idx])

        self.color_K = bg_calib['color_intrinsic']
        self.depth_K = bg_calib['depth_intrinsic']

    def run(self):
        scene = bpy.context.scene

        output_types = self.params['output_types']

        restart_blender_scene()

        self.log_message("Initializing scene")
        fg_humans, bpy_camera_obj = self.init_scene()

        orig_cam_loc = bpy_camera_obj.location.copy()

        smpl_DoF = 10 # only pick the top DoF for the creations, maximum 10

        # for each clipsize'th frame in the sequence
        random_zrot = 0
        reset_loc = False
        batch_it = 0
        random_zrot = 2*np.pi*np.random.rand()

        bpy_camera_obj.animation_data_clear()

        # set for optical flow
        for part, material in fg_humans[0].materials.items():
            material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

        # set up random light
        shading_params = .7 * (2 * np.random.rand(9) - 1)
        shading_params[0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
        shading_params[1] = -.7 * np.random.rand()

        # spherical harmonics material needs a script to be loaded and compiled
        spherical_harmonics = []
        for mname, material in fg_humans[0].materials.items():
            spherical_harmonics.append(material.node_tree.nodes['Script'])
            spherical_harmonics[-1].filepath = self.sh_dst
            spherical_harmonics[-1].update()

        for ish, coeff in enumerate(shading_params):
            for sc in spherical_harmonics:
                sc.inputs[ish+1].default_value = coeff

        ''' -------------------- LOOP TO CREATE 3D ANIMATION ---------------- '''
        # create a keyframe animation with pose, translation, blendshapes and camera motion
        for frame_idx in range(0, len(self.cam_poses)):

            scene.frame_set(frame_idx)

            bpy_camera_obj.matrix_world = world_to_blender(Matrix(self.cam_poses[frame_idx]))

            bpy_camera_obj.keyframe_insert('location', frame=frame_idx)
            bpy_camera_obj.keyframe_insert('rotation_euler', frame=frame_idx)

            # apply the translation, pose and shape to the character
            body_data_index = frame_idx * self.fg_stride
            for idx in range(self.fg_total_number):
                pose, trans = fg_humans[idx].apply_Rt_body_shape(body_data_index, frame_idx)

            scene.update()

        ''' ---------------------- LOOP TO RENDER  ------------------------- '''
        # iterate over the keyframes and render
        for frame_idx in range(0, len(self.cam_poses)):

            scene.frame_set(frame_idx)

            scene.render.use_antialiasing = False
            scene.render.filepath = join(self.tmp_path, 'Image%04d.png' % frame_idx)

            self.log_message("Rendering frame {:d} for scene {:s} stride {:d}".format(frame_idx, self.scene_name, self.scene_stride))

            # Render
            bpy.ops.render.render(write_still=True)

            for idx in range(self.fg_total_number):
                fg_humans[idx].reset_pose()

    # creation of the spherical harmonics material, using an OSL script
    def create_shader_material(self, tree, sh_path, texture):
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        uv = tree.nodes.new('ShaderNodeTexCoord')
        uv.location = -800, 400

        uv_xform = tree.nodes.new('ShaderNodeVectorMath')
        uv_xform.location = -600, 400
        uv_xform.inputs[1].default_value = (0, 0, 1)
        uv_xform.operation = 'AVERAGE'

        # for pair in self.clothing_names:
        cloth_img = bpy.data.images.load(texture)
        uv_im = tree.nodes.new('ShaderNodeTexImage')
        uv_im.location = -400, 400
        uv_im.image = cloth_img

        rgb = tree.nodes.new('ShaderNodeRGB')
        rgb.location = -400, 200

        script = tree.nodes.new('ShaderNodeScript')
        script.location = -230, 400
        script.mode = 'EXTERNAL'
        #using the same file from multiple jobs may causes white texture
        script.filepath = sh_path #'sphere_harmonics_lighting/sh.osl' 
        script.update()

        # the emission node makes it independent of the scene lighting
        emission = tree.nodes.new('ShaderNodeEmission')
        emission.location = -60, 400

        mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
        mat_out.location = 110, 400

        tree.links.new(uv.outputs[2], uv_im.inputs[0])
        tree.links.new(uv_im.outputs[0], script.inputs[0])
        tree.links.new(script.outputs[0], emission.inputs[0])
        tree.links.new(emission.outputs[0], mat_out.inputs[0])

    def init_scene(self):
        """ Initialize the blender scene and renderer
        """

        # delete the default cube (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete(use_global=False)

        # set camera properties and initial position
        bpy.ops.object.select_all(action='DESELECT')
        bpy_camera_obj = bpy.data.objects['Camera']
        bpy_scene = bpy.context.scene
        bpy_scene.objects.active = bpy_camera_obj

        bpy_camera_obj.matrix_world = world_to_blender(Matrix(self.cam_poses[0]))

        K = [self.color_K[0][0], self.color_K[1][1], self.color_K[0,2], self.color_K[1,2]]
        set_intrinsic(K, bpy_camera_obj, bpy_scene, self.params['height'], self.params['width'])

        bpy_render = bpy_scene.render

        ##### set cycles and cuda
        cycles_preferences = bpy.context.user_preferences.addons['cycles'].preferences
        # bpy_scene.cycles.device = 'GPU'
        bpy_render.use_overwrite = False
        bpy_render.use_placeholder = True
        # cycles_preferences.compute_device_type = "CUDA"

        bpy_scene.cycles.film_transparent = True
        bpy_render.layers["RenderLayer"].use_pass_vector = self.params['output_types']['gtflow']
        bpy_render.layers["RenderLayer"].use_pass_normal = self.params['output_types']['normal']        
        bpy_render.layers["RenderLayer"].use_pass_z = self.params['output_types']['depth']
        bpy_render.layers['RenderLayer'].use_pass_emit   = False
        bpy_scene.render.layers['RenderLayer'].use_pass_material_index  = True

        # set render size
        bpy_render.resolution_x = self.params['width']
        bpy_render.resolution_y = self.params['height']
        bpy_render.resolution_percentage = 100
        # bpy_scene.render.image_settings.file_format = 'PNG'
        bpy_render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        bpy_render.image_settings.color_mode = 'RGBA'

        # set the render parameters
        bpy_render.use_raytrace = False
        bpy_render.tile_x = 512
        bpy_render.tile_y = 512
        bpy_scene.cycles.max_bounces = 8
        bpy_scene.cycles.samples = 64

        fg_humans = []
        fg_object_stride = int(len(self.cam_poses) / self.fg_total_number)

        W, H = self.params['width'], self.params['height']
        inv_K33 = np.linalg.inv(self.color_K[:3, :3])
        near_frustum_threshold = 0.5
        for idx in range(self.fg_total_number):
            # set the foreground to be at certain footages in the scene
            # the foreground objects will be equally distributed along the camera pose
            choose_pose = self.cam_poses[idx*fg_object_stride]

            # set the position of foreground objects
            far_frustum_threshold = 1e3
            while True:
                np.random.seed()
                choose_u = int(np.random.uniform(128, W-128))
                choose_v = int(np.random.uniform(128, H-128))
                loc_2d = [choose_u, choose_v, 1.0]

                bg_depth = imread(self.bg_depth_files[idx]) / 1.0e3
                # if it samples value, it will restart sampling
                far_frustum_threshold = min(bg_depth[choose_v, choose_u] - 0.3, 3.0)

                if far_frustum_threshold > near_frustum_threshold: break

            # load the distance of background map
            distance = np.random.uniform(near_frustum_threshold, far_frustum_threshold)
            loc_3d = inv_K33.dot(loc_2d) * distance
            loc_3d *= np.array([1, -1, -1])

            # randomly choose a camera pose from the scene, and put a human body in the scene.
            cam_pose = world_to_blender(Matrix(choose_pose))
            # set up the material for the object
            material = bpy.data.materials.new(name='Material'+str(idx))
            # material = bpy.data.materials['Material']
            material.use_nodes = True
            self.create_shader_material(material.node_tree, self.sh_dst, self.clothing_names[idx][1])

            # randomly generate action number
            fg_human = SMPL_Body(self.smpl_data, self.clothing_names[idx][0], cam_pose, material, idx, anchor_location3d=loc_3d)

            fg_human.obj.active_material = material
            fg_humans.append(fg_human)

        return (fg_humans, bpy_camera_obj)

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate synth dataset images')
    parser.add_argument('--fg_density', type=int, default=0, help='Every N frames we will load an object into the scene.')
    parser.add_argument('--fg_stride', type=int, default=0, help='The number of stride when we load')
    parser.add_argument('--bg_scene', type=str, default='None', help='The scene trajectory to be used as background')
    parser.add_argument('--bg_stride', type=int, default=0, help='The number of background stride')
    parser.add_argument('--bg_start', type=int, default=-1, help='The start frame in the background trajectory')
    parser.add_argument('--bg_end', type=int, default=-1, help='The end frame in the background trajectory')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    fg_density, fg_stride, bg_scene, bg_stride, bg_start, bg_end = None, None, None, None, None, None
    if args.fg_density != 0 :
        fg_density = args.fg_density
    if args.fg_stride != 0:
        fg_stride = args.fg_stride
    if args.bg_scene != 'None':
        bg_scene = args.bg_scene
    if args.bg_stride != 0:
        bg_stride = args.bg_stride
    if args.bg_start != -1:
        bg_start = args.bg_start
    if args.bg_end != -1:
        bg_end = args.bg_end

    sg = DynamicHumanRender(fg_density, fg_stride, bg_scene, bg_stride, bg_start, bg_end)

    sg.run()

