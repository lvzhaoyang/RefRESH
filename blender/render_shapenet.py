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

import sys, os, time, random
import os.path as osp
import numpy as np
import bpy

from numpy.random import uniform as rand_uniform
from pickle import load, dump

sys.path.insert(0, ".")
import utils as bpy_utils
import shapenet_ids as sp
from io_utils import load_file

class ShapeNetRender:
    """ Render the shapenet obj given the camera viewpoints
    @todo: replace it by directly loading the taxonomy.json file
    """

    def __init__(self, root_path, post_fix='', object_id = ''):
        """ The path of shapeNet folder
        """
        self.start_time = time.time()

        self.log_message("Importing ShapeNet models and set up")

        self.params = load_file('configs/shapenet_config', 'SHAPENET_MODELS')

        # shapenet_taxonomy = osp.join(root_path, 'taxonomy.json')

        shape_dict = dict(sp.shape_synset_name_pairs)

        if object_id == '':
            shape_ids = self.params['shape_net_ids']

            self.shape_list = []
            for shape_id in shape_ids:
                # randomly choose obj index from the list
                synset = shape_dict[shape_id]
                print("Choose shape category {:}: {:}".format(shape_id, synset))
                shape_path = osp.join(root_path, shape_id)

                self.shape_list += self.load_category_shape_list(shape_path)
        else:
            synset = shape_dict[object_id]
            print("Choose shape category {:}: {:}".format(object_id, synset))
            shape_path = osp.join(root_path, object_id)
            self.shape_list = self.load_category_shape_list(shape_path)

            self.params['tmp_path'] = osp.join(self.params['tmp_path'], synset)
            self.params['output_path'] = osp.join(self.params['output_path'], synset)

        # where all the openexr files are written to
        self.tmp_path = osp.join(self.params['tmp_path'], post_fix)

        self.init_render_settings()

        self.init_scene()

    def load_category_shape_list(self, path):
        """ load shapes from one category
        """
        shape_md5_list = os.listdir(path)
        shape_list = []
        for shape_md5 in shape_md5_list:
            # path = osp.join(path, shape_md5, 'images/')
            if(osp.isdir(path)):
                shape_list.append((shape_md5, osp.join(path, shape_md5, 'model.obj')))
            else:
                print("The following path is missing: {:}".format(path))
        return shape_list

    def run_rendering(self):
        """ Run rendering script
        """

        info = {
            'camera_pose':  [],
            'object_pose':  {},
            'object_3D_box':{}
        }

        for name in self.object_list:
            info['object_pose'][name] = []
            info['object_3D_box'][name]=[]

        get_real_frame = lambda ifr: ifr
        bpy_scene = bpy.context.scene
        bpy_object= bpy.data.objects
        bpy_camera= bpy_object['Camera']

        num_views = self.params['trajectory']['views']

        ''' ---------------------- LOOP TO RENDER  ------------------------- '''
        for frame_idx in range(0, num_views):
            bpy_scene.frame_set(get_real_frame(frame_idx))

            bpy_scene.render.filepath = osp.join(self.tmp_path, 'Image%04d.exr' % get_real_frame(frame_idx))

            self.log_message("Rendering frame %d" % frame_idx)
            # Render
            bpy.ops.render.render(write_still=True)
            # camera pose
            info['camera_pose'].append(
                np.array(bpy_utils.blender_to_world(bpy_camera.matrix_world)) )

            for name in self.object_list:
                # the object pose
                info['object_pose'][name].append(
                    np.array(bpy_utils.blender_to_world(bpy_object[name].matrix_world)).copy()
                )
                # The corners are in allocentric coordinate.
                corners = np.array(bpy_object[name].bound_box)
                info['object_3D_box'][name].append(corners.copy())

        ''' ---------------------- SAVE 3D OUTPUT  ------------------------- '''
        dataset_path = osp.join(self.tmp_path, 'info.pkl')
        with open(dataset_path, 'wb') as output:
            dump(info, output)

    def init_render_settings(self):
        """ Set up the the blender (cycles) render engine
        """
        self.log_message("Setup Blender (Cycles) Render Engine")

        bpy_scene = bpy.context.scene
        bpy_render = bpy_scene.render

        # Turn off the cycles rendering system
        # bpy_scene.cycles.shading_system = True

        bpy_render.use_overwrite = False
        bpy_render.use_placeholder = True
        bpy_render.use_antialiasing = True

        bpy_render.layers["RenderLayer"].use_pass_vector = self.params['output_types']['gtflow']
        bpy_render.layers["RenderLayer"].use_pass_normal = self.params['output_types']['normal']
        bpy_render.layers["RenderLayer"].use_pass_object_index = self.params['output_types']['segm']
        bpy_render.layers["RenderLayer"].use_pass_z = self.params['output_types']['depth']
        bpy_render.layers['RenderLayer'].use_pass_emit   = False

        # set render size
        bpy_render.resolution_x = self.params['width']
        bpy_render.resolution_y = self.params['height']

        bpy_render.resolution_percentage = 100
        # bpy_render.image_settings.file_format = 'PNG'
        bpy_render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        bpy_render.image_settings.color_mode = 'RGBA'

    def init_scene(self):
        """ Initialize the objects in the scene
        """

        self.log_message("Initialize the scene")

        # change the cube to be the default 3D background (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy_cube = bpy.data.objects['Cube']
        bpy_cube.select = True
        bs = self.params['box_scale']
        bpy_cube.scale = bs, bs, bs
        # bpy.ops.object.delete()
#
        # bpy_cube.join_uvs()
        bpy.ops.uv.smart_project()
        bpy.context.scene.objects.active = bpy_cube

        # create background material map
        bg_material_uv = self.make_material('Random',
            diffuse = [rand_uniform(0.5,1), rand_uniform(0.5,1), rand_uniform(0.5,1)],
            specular= [rand_uniform(0,0.5), rand_uniform(0,0.5), rand_uniform(0,0.5)],
            alpha = 1.0)
        bg_material_uv.type = 'SURFACE'

        bg_material_path = self.params['background_material_path']
        bg_material_images = os.listdir(bg_material_path)
        bg_material_img = random.choice(bg_material_images)
        uvimg = bpy.data.images.load(osp.join(bg_material_path, bg_material_img))
        bg_texture = bpy.data.textures.new('background_texture', 'IMAGE')
        bg_texture.image = uvimg
        bpy_cube.material_slots[0].material = bg_material_uv
        slot =  bpy.context.object.active_material.texture_slots.add()
        slot.texture = bg_texture

        # load a random number of objects into the scene
        num_obj = self.params['num_obj']
        shape_ids = np.random.randint(0, len(self.shape_list) - 1, num_obj)

        view_params = self.params['trajectory']['views']
        obj_motion_t_max = self.params['obj_motions']['t_max']
        obj_motion_r_max = self.params['obj_motions']['r_max']

        bpy_objects = bpy.context.scene.objects

        self.object_list = []
        obj_index = 1
        for shape_id in shape_ids:
            existing_objects = bpy_objects.keys()

            obj, obj_path = self.shape_list[shape_id]
            bpy.ops.import_scene.obj(filepath=obj_path)
            # combine meshes of the imported model (which are composed of different parts)
            for obj_parts in bpy_objects:
                if obj_parts.name not in existing_objects:
                    obj_parts.select = True
                    bpy.context.scene.objects.active = obj_parts
                else:
                    obj_parts.select = False

            bpy.ops.object.join()
            if len(bpy.context.selected_objects) < 1:
                continue

            current_obj = bpy.context.selected_objects[0]
            current_obj.name = 'Model_{:}'.format(obj_index)
            current_obj.pass_index = obj_index
            obj_index += 1

            self.object_list.append(current_obj.name)

            # current_obj.game.physics_type = 'RIGID_BODY'
            # current_obj.game.use_collision_bounds = 1

            # set the obj pose (to be random)
            for frame_idx in range(0, view_params+1, 10):
                azimuth_deg = rand_uniform(-obj_motion_r_max, obj_motion_r_max)
                theta_deg = rand_uniform(-obj_motion_r_max, obj_motion_r_max)
                radius = rand_uniform(0, obj_motion_t_max)
                current_obj.location = bpy_utils.allocentric_pose(radius, azimuth_deg, theta_deg)

                current_obj.rotation_mode = 'QUATERNION'
                yaw_deg  = rand_uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                pitch_deg= rand_uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                roll_deg = rand_uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                current_obj.rotation_quaternion = bpy_utils.ypr2quaternion(yaw_deg, pitch_deg, roll_deg)

                current_obj.keyframe_insert('location', frame=frame_idx)
                current_obj.keyframe_insert('rotation_quaternion', frame=frame_idx)

        # self.set_environment_lighting()

        # set four point light inside of the cube
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)
        # bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(0, 0, 0))
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(rand_uniform(-bs, 0), rand_uniform(0, 0), 0))
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(rand_uniform(0, bs), rand_uniform(0, 0), 0))
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(rand_uniform(0, 0), rand_uniform(-bs, 0), 0))
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(rand_uniform(0, 0), rand_uniform(0, bs), 0))

        self.set_camera_parameters()

    def set_camera_parameters(self):
        """ Set the blender camera intrinsic and extrinsic parameters
        """
        bpy.ops.object.select_all(action='DESELECT')
        bpy_camera = bpy.data.objects['Camera']
        bpy_scene = bpy.context.scene
        bpy_scene.objects.active = bpy_camera

        pinhole_params = self.params['pinhole']
        K = [pinhole_params['fx'], pinhole_params['fy'], pinhole_params['cx'], pinhole_params['cy']]
        bpy_utils.set_intrinsic(K, bpy_camera, bpy_scene, self.params['height'], self.params['width'])

        # set the camera trajectory according to the ground truth trajectory
        traj_settings = self.params['trajectory']
        if traj_settings['type'] == 'ORBIT':
            cam_constraint = bpy_camera.constraints.new(type='TRACK_TO')
            cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            cam_constraint.up_axis = 'UP_Y'
            bpy_track_origin = bpy.data.objects.new("Empty", None)
            bpy_track_origin = bpy_utils.parent_obj_to_camera(bpy_camera, bpy_scene, bpy_track_origin)
            cam_constraint.target = bpy_track_origin

            view_params = traj_settings['views']
            # the distance of the camera to the center of the object
            radius = traj_settings['radius']
            keyframe_deg = 360 / view_params
            azimuth_deg = 0
            elevation_deg = 0
            theta_deg = 0
            for frame_idx in range(view_params):

                bpy_scene.frame_set(frame_idx)
                # set the camera extrinsic
                bpy_camera.location = bpy_utils.allocentric_pose(radius, azimuth_deg, theta_deg)

                bpy_camera.keyframe_insert('location', frame=frame_idx)
                bpy_camera.keyframe_insert('rotation_euler', frame=frame_idx)

                bpy_scene.update()
                azimuth_deg += keyframe_deg

        elif traj_settings['type'] == 'STATIC':
            cam_constraint = bpy_camera.constraints.new(type='TRACK_TO')
            cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            cam_constraint.up_axis = 'UP_Y'
            bpy_track_origin = bpy.data.objects.new("Empty", None)
            bpy_track_origin = bpy_utils.parent_obj_to_camera(bpy_camera, bpy_scene, bpy_track_origin)
            cam_constraint.target = bpy_track_origin

            # the distance of the camera to the center of the object
            radius = traj_settings['radius']
            azimuth_deg = 360 * rand_uniform(0,1) # random generate view-point
            elevation_deg = 0
            theta_deg = 0
            bpy_camera.location = bpy_utils.allocentric_pose(radius, azimuth_deg, theta_deg)
            bpy_camera.keyframe_insert('location', frame=0)
            bpy_camera.keyframe_insert('rotation_euler', frame=0)
            bpy_scene.update()

        elif traj_settings['type'] == 'Existing':

            for frame_idx in range(0, self.total_num):

                bpy_scene.frame_set(frame_idx)

                bpy_camera.matrix_world = world_to_blender(Matrix(self.cam_poses[frame_idx]))
                bpy_camera.keyframe_insert('location', frame=frame_idx)
                bpy_camera.keyframe_insert('rotation_euler', frame=frame_idx)

                bpy_scene.update()

    def set_environment_lighting(self):
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)

        # set environment lighting
        light_settings = bpy.context.scene.world.light_settings
        light_settings.use_environment_light = True
        light_settings.environment_energy = rand_uniform(self.params['env_light_min'],
            self.params['env_light_max'])
        light_settings.environment_color = 'PLAIN'

    def make_material(self, name, diffuse, specular, alpha):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = diffuse
        mat.diffuse_shader = 'LAMBERT'
        mat.diffuse_intensity = 0.8
        mat.specular_color = specular
        mat.specular_shader = 'BLINN'
        mat.specular_intensity = 0.3
        mat.alpha = alpha
        mat.ambient = 1
        return mat

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

def reset_blend():
    bpy.ops.wm.read_factory_settings()

if __name__ == '__main__':

    shapenet_path = '/is/sg/zlv/data-avg/datasets/ShapeNetCore.v1'

    import argparse
    parser = argparse.ArgumentParser(description='Generate 3D shapes')
    parser.add_argument('--shape_id', type=str, default='None',
        help='the shapenet id')
    parser.add_argument('--seq_num', type=int, default=1,
        help='the number of sequences being generated')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--object_id', type=str, default='')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    # from parse_shapenet import ShapeNetSceneParser

    for idx in range(args.seq_num):
        post_fix = "{:06d}".format(idx+args.start_index)
        shape_net_render = ShapeNetRender(shapenet_path, post_fix, args.object_id)
        shape_net_render.run_rendering()
        # shapenet_parser = ShapeNetSceneParser(post_fix)
        # shapenet_parser.run()
        reset_blend()
