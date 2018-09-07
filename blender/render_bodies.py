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

import sys, os, math, time
import numpy as np

from os import getenv, remove
from os.path import join, dirname, realpath, exists
from glob import glob
from pickle import load
# from random import choice, seed, random

import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.insert(0, ".")
from motion_surreal import *

from utils import world_to_blender, set_intrinsic

def create_directory(target_dir):
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

def restart_blender_scene():
    for obj in bpy.data.objects.values():
        obj.select=False
    bpy.context.scene.objects.active = None

class SceneGeneration:

    start_time = None

    frame_per_clip = 200 # We will generate 200 poses per clip

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

    def __init__(self, fg_stride, fg_number):
        '''
        Foreground action stride. You can make it random for each object
        '''
        self.fg_stride = fg_stride
        self.fg_total_number = fg_number

        self.start_time = time.time()

        self.params = io_utils.load_file('body_config', 'SYNTH_HUMAN')

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
        seed()

        idx_info = load(open("pkl/idx_info.pickle", 'rb'))
        # random load foreground indices
        idx_info_len = len(idx_info)
        fg_indices = [int(idx_info_len*random()) for i in range(idx_info_len)]

        self.fg_indices_info = []
        for idx in fg_indices:
            self.fg_indices_info.append(idx_info[idx])

        #######################################################################
        self.log_message("Loading the smpl data")
        smpl_data_folder = self.params['smpl_data_folder']
        smpl_data_filename = self.params['smpl_data_filename']
        self.smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

        ######################################################################
        self.log_message('Set up background information')
        self.init_camera()

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
        self.init_directories()

        # >> don't use random generator before this point <<
        # initialize RNG with seeds from sequence id ToDo: not sure whether still useful or not
        import hashlib
        s = "synth_data:{:d}".format(fg_stride)
        seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        self.log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
        np.random.seed(seed_number)

        output_types = self.params['output_types']
        if(output_types['vblur']):
            vblur_factor = np.random.normal(0.5, 0.5)
            self.params['vblur_factor'] = vblur_factor

        #######################################################################
        # grab clothing names
        self.log_message('Set up blender node')
        self.res_paths = self.create_composite_nodes(scene.node_tree)

    def init_directories(self):
        '''
        how the data will be saved
        '''

        folder_name = 'bodies_output'
        tmp_path = self.params['tmp_path']
        tmp_path = join(tmp_path, folder_name)

        self.tmp_path = tmp_path
        print('The blender output will be written to {:s}'.format(self.tmp_path))
        if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
            os.system('rm -rf %s' % tmp_path)
        rgb_vid_filename = folder_name + ".mp4"

        create_directory(tmp_path)

        # create copy-spher.harm. directory if not exists
        sh_dir = join(tmp_path, 'spher_harm')
        create_directory(sh_dir)
        self.sh_dst = join(sh_dir, 'sh_sphere.osl')
        os.system('cp spher_harm/sh.osl {:s}'.format(self.sh_dst))

        self.rgb_path = join(tmp_path, 'rgb_video.mp4')

    def init_camera(self):
        '''
        Currently no camera poses are loaded. You can random set the camera trajectory to render multiple images at one time.
        Leave to Abhijit as an option TODO.
        '''
        # load the pickle file generated from background rendering

        self.cam_poses = []
        self.cam_poses.append(np.eye(4))

        # set or load the camera intrinsic here
        K = [528.871, 528.871, 320, 240]
        bpy_camera_obj = bpy.data.objects['Camera']
        bpy_scene = bpy.context.scene

        set_intrinsic(K, bpy_camera_obj, bpy_scene, self.params['height'], self.params['width'])

        self.K = np.eye(3)
        self.K[0,0] = K[0]
        self.K[1,1] = K[1]
        self.K[0,2] = K[2]
        self.K[1,2] = K[3]

    def run(self):
        # time logging

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

        ''' -------------------- LOOP TO CREATE 3D ANIMATION '''
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
            scene.render.filepath = join(self.rgb_path, 'Image%04d.png' % frame_idx)

            self.log_message("Rendering frame {:d}".format(frame_idx))

            # disable render output
            logfile = '/dev/null'
            open(logfile, 'a').close()
            old = os.dup(1)
            sys.stdout.flush()
            os.close(1)
            os.open(logfile, os.O_WRONLY)

            # Render
            bpy.ops.render.render(write_still=True)

            # disable output redirection
            os.close(1)
            os.dup(old)
            os.close(old)

            for idx in range(self.fg_total_number):
                fg_humans[idx].reset_pose()

    def create_composite_nodes(self, tree, img=None):
        '''
        Create the different passes for blender rendering.
        Note: refer to blender render passes for all the relevant information:
        https://docs.blender.org/manual/en/dev/render/blender_render/settings/passes.html

        We use cycles engine in our renderng setting: https://docs.blender.org/manual/en/dev/render/cycles/settings/scene/render_layers/passes.html
        '''
        res_paths = {k:join(self.tmp_path, k) for k in self.params['output_types'] if self.params['output_types'][k]}

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create node for foreground image
        layers = tree.nodes.new('CompositorNodeRLayers')
        layers.location = -300, 400

        if(self.params['output_types']['vblur']):
        # create node for computing vector blur (approximate motion blur)
            vblur = tree.nodes.new('CompositorNodeVecBlur')
            vblur.factor = params['vblur_factor']
            vblur.location = 240, 400

            # create node for saving output of vector blurred image
            vblur_out = tree.nodes.new('CompositorNodeOutputFile')
            vblur_out.format.file_format = 'PNG'
            vblur_out.base_path = res_paths['vblur']
            vblur_out.location = 460, 460

        # create node for the final output
        composite_out = tree.nodes.new('CompositorNodeComposite')
        composite_out.location = 240, 30

        # create node for saving depth
        if(self.params['output_types']['depth']):
            depth_out = tree.nodes.new('CompositorNodeOutputFile')
            depth_out.location = 40, 700
            depth_out.format.file_format = 'OPEN_EXR'
            depth_out.base_path = res_paths['depth']

        # create node for saving normals
        if(self.params['output_types']['normal']):
            normal_out = tree.nodes.new('CompositorNodeOutputFile')
            normal_out.location = 40, 600
            normal_out.format.file_format = 'OPEN_EXR'
            normal_out.base_path = res_paths['normal']

        # create node for saving foreground image
        if(self.params['output_types']['fg']):
            fg_out = tree.nodes.new('CompositorNodeOutputFile')
            fg_out.location = 170, 600
            fg_out.format.file_format = 'PNG'
            fg_out.base_path = res_paths['fg']

        # create node for saving ground truth flow
        if(self.params['output_types']['gtflow']):
            gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
            gtflow_out.location = 40, 500
            gtflow_out.format.file_format = 'OPEN_EXR'
            gtflow_out.base_path = res_paths['gtflow']

        # create node for saving segmentation
        if(self.params['output_types']['segm']):
            segm_out = tree.nodes.new('CompositorNodeOutputFile')
            segm_out.location = 40, 400
            segm_out.format.file_format = 'OPEN_EXR'
            segm_out.base_path = res_paths['segm']

        if(self.params['output_types']['vblur']):
            tree.links.new(layers.outputs['Image'], vblur.inputs[0])                # apply vector blur on the bg+fg image,
            tree.links.new(layers.outputs['Depth'], vblur.inputs[1])       #   using depth,
            tree.links.new(layers.outputs['Vector'], vblur.inputs[2])       #   and flow.
            tree.links.new(vblur.outputs[0], vblur_out.inputs[0])          # save vblurred output

        if(self.params['output_types']['fg']):
            tree.links.new(layers.outputs['Image'], fg_out.inputs[0])      # save fg
        if(self.params['output_types']['depth']):
            tree.links.new(layers.outputs['Depth'], depth_out.inputs[0])   # save depth
        if(self.params['output_types']['normal']):
            tree.links.new(layers.outputs['Normal'], normal_out.inputs[0]) # save normal
        if(self.params['output_types']['gtflow']):
            tree.links.new(layers.outputs['Vector'], gtflow_out.inputs[0])  # save ground truth flow
        if(self.params['output_types']['segm']):
            # IndexMA: get access to alpha value per object per mask
            # https://docs.blender.org/manual/en/dev/compositing/types/converter/id_mask.html
            tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation

        return(res_paths)

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
        script.filepath = sh_path #'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
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
        '''init_scene
        Initialize the blender scene environment
        '''
        # TODO: add the scene loading functions
        # may also need to add the camera sequence here

        # assign the existing spherical harmonics material
        #fg_obj.active_material = bpy.data.materials['Material']

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

        ##### set cycles and cuda
        cycles_preferences = bpy.context.user_preferences.addons['cycles'].preferences
        bpy_scene.render.use_overwrite = False
        bpy_scene.render.use_placeholder = True
        cycles_preferences.compute_device_type = "CUDA"

        bpy_scene.cycles.film_transparent = True
        bpy_scene.render.layers["RenderLayer"].use_pass_vector = True
        bpy_scene.render.layers["RenderLayer"].use_pass_normal = True
        bpy_scene.render.layers['RenderLayer'].use_pass_emit   = True
        bpy_scene.render.layers['RenderLayer'].use_pass_material_index  = True

        # set render size
        bpy_scene.render.resolution_x = self.params['width']
        bpy_scene.render.resolution_y = self.params['height']
        bpy_scene.render.resolution_percentage = 100
        bpy_scene.render.image_settings.file_format = 'PNG'

        # set the render parameters
        bpy_scene.render.use_raytrace = False
        bpy_scene.render.tile_x = 512
        bpy_scene.render.tile_y = 512
        bpy_scene.cycles.max_bounces = 8
        bpy_scene.cycles.samples = 64

        W, H = self.params['width'], self.params['height']
        fg_humans = []

        for idx in range(self.fg_total_number):

            # randomly set the camera pose here, leave to Abhijit to set the foreground poses
            np.random.seed()
            loc_2d = [np.random.uniform(0, W), np.random.uniform(0, H), 1.0]
            distance = np.random.uniform(1, 20)
            # Not sure how you want to normalize it.
            loc_3d = np.linalg.inv(self.K).dot(loc_2d) * distance
            # transform coordinate to blender
            loc_3d *= np.array([1, -1, -1])
            cam_pose = world_to_blender(Matrix(self.cam_poses[0]))

            # set up the material for the object
            material = bpy.data.materials.new(name='Material'+str(idx))
            # material = bpy.data.materials['Material']
            material.use_nodes = True
            self.create_shader_material(material.node_tree, self.sh_dst, self.clothing_names[idx][1])

            # randomly generate action number
            gender = self.clothing_names[idx][0]
            fg_human = SMPL_Body(self.smpl_data, gender, cam_pose, material, idx, anchor_location3d=loc_3d)
            fg_human.obj.active_material = material
            fg_humans.append(fg_human)

        return fg_humans, bpy_camera_obj

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Generate synth dataset images')
    parser.add_argument('--fg_stride', type=int, default=0, help='The number of stride when we load')
    parser.add_argument('--fg_number', type=int, default=0, help='The total number of foreground bodies')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    fg_stride, fg_number = None, None
    if args.fg_stride != 0:
        fg_stride = args.fg_stride
    if args.fg_number != 0:
        fg_number = args.fg_number

    sg = SceneGeneration(fg_stride, fg_number)

    sg.run()

    # sys.exit()
