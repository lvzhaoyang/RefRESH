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

import sys, os, time
import os.path as osp
import numpy as np
import bpy

sys.path.insert(0, ".")
import utils as bpy_utils
from io_utils import load_file

class ShapeNetRender:
    """ Render the shapenet obj given the camera viewpoints
    @todo: replace it by directly loading the taxonomy.json file
    """
    shape_synset_name_pairs = [('02691156', 'aeroplane'),
    							('02747177', ''),
    							('02773838', ''),
    							('02801938', ''),
    							('02808440', ''),
    							('02818832', ''),
    							('02828884', ''),
    							('02834778', 'bicycle'),
    							('02843684', ''),
    							('02858304', 'boat'),
    							('02871439', ''),
    							('02876657', 'bottle'),
    							('02880940', ''),
    							('02924116', 'bus'),
    							('02933112', ''),
    							('02942699', ''),
    							('02946921', ''),
    							('02954340', ''),
    							('02958343', 'car'),
    							('02992529', ''),
    							('03001627', 'chair'),
    							('03046257', ''),
    							('03085013', ''),
    							('03207941', ''),
    							('03211117', 'tvmonitor'),
    							('03261776', ''),
    							('03325088', ''),
    							('03337140', ''),
    							('03467517', ''),
    							('03513137', ''),
    							('03593526', ''),
    							('03624134', ''),
    							('03636649', ''),
    							('03642806', ''),
    							('03691459', ''),
    							('03710193', ''),
    							('03759954', ''),
    							('03761084', ''),
    							('03790512', 'motorbike'),
    							('03797390', ''),
    							('03928116', ''),
    							('03938244', ''),
    							('03948459', ''),
    							('03991062', ''),
    							('04004475', ''),
    							('04074963', ''),
    							('04090263', ''),
    							('04099429', ''),
    							('04225987', ''),
    							('04256520', 'sofa'),
    							('04330267', ''),
    							('04379243', 'diningtable'),
    							('04401088', ''),
    							('04460130', ''),
    							('04468005', 'train'),
    							('04530566', ''),
    							('04554684', '')];

    def __init__(self, root_path, post_fix=''):
        """ The path of shapeNet folder
        """
        self.start_time = time.time()

        self.log_message("Importing ShapeNet models and set up")

        self.params = load_file('configs/shapenet_config', 'SHAPENET_MODELS')

        # shapenet_taxonomy = osp.join(root_path, 'taxonomy.json')

        shape_ids = self.params['shape_net_ids']

        self.shape_list = []
        for shape_id in shape_ids:
            # randomly choose obj index from the list
            shape_dict = dict(self.shape_synset_name_pairs)
            synset = shape_dict[shape_id]
            print("Choose shape category {:}: {:}".format(shape_id, synset))
            shape_path = osp.join(root_path, shape_id)

            self.shape_list += self.load_category_shape_list(shape_path)

        # where all the openexr files are written to
        self.tmp_path = osp.join(self.params['tmp_path'], post_fix)

        self.init_scene()

        self.init_render_settings()

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
                    np.array(bpy_utils.blender_to_world(bpy_object[name].matrix_world))
                )
                # The corners are in allocentric coordinate.
                corners = np.array(bpy_object[name].bound_box)
                info['object_3D_box'][name].append(corners)

    def init_render_settings(self):
        """ Set up the the blender (cycles) render engine
        """
        self.log_message("Setup Blender Cycles Render Engine")

        bpy_scene = bpy.context.scene
        bpy_render = bpy_scene.render

        bpy_scene.cycles.shading_system = True

        bpy_render.use_overwrite = False
        bpy_render.use_placeholder = True
        bpy_render.use_antialiasing = True

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

    def init_scene(self):
        """ Initialize the objects in the scene
        """

        self.log_message("Initialize the scene")

        # delete the default cube (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete()

        # load a random number of objects into the scene
        num_obj = self.params['num_obj']
        shape_ids = np.random.randint(0, len(self.shape_list) - 1, num_obj)

        view_params = self.params['trajectory']['views']
        obj_motion_t_max = self.params['obj_motions']['t_max']
        obj_motion_r_max = self.params['obj_motions']['r_max']

        self.object_list = []
        for shape_id in shape_ids:
            obj, obj_path = self.shape_list[shape_id]
            bpy.ops.import_scene.obj(filepath=obj_path)

            # combine meshes of the imported model (which are composed of different parts)
            for obj_parts in bpy.context.scene.objects:
                if obj_parts.type == 'MESH' and obj_parts.name[:5] != 'Model':
                    obj_parts.select = True
                    bpy.context.scene.objects.active = obj_parts
                else:
                    obj_parts.select = False

            bpy.ops.object.join()
            current_obj = bpy.context.selected_objects[0]
            current_obj.name = 'Model_{:}'.format(obj)

            self.object_list.append(current_obj.name)

            # set the obj pose (to be random)
            for frame_idx in range(0, view_params, 10):
                azimuth_deg = np.random.uniform(-obj_motion_r_max, obj_motion_r_max)
                theta_deg = np.random.uniform(-obj_motion_r_max, obj_motion_r_max)
                radius = np.random.uniform(0, obj_motion_t_max)
                current_obj.location = bpy_utils.allocentric_pose(radius, azimuth_deg, theta_deg)

                current_obj.rotation_mode = 'QUATERNION'
                yaw_deg = np.random.uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                pitch_deg=np.random.uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                roll_deg =np.random.uniform(-obj_motion_r_max, obj_motion_r_max) / 180
                current_obj.rotation_quaternion = bpy_utils.ypr2quaternion(yaw_deg, pitch_deg, roll_deg)

                current_obj.keyframe_insert('location', frame=frame_idx)
                current_obj.keyframe_insert('rotation_quaternion', frame=frame_idx)

        self.set_environment_lighting()

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

        elif traj_settings['type'] == 'LINE':
            pass

        elif traj_settings['type'] == 'Existing':

            for frame_idx in range(0, self.total_num):

                bpy_scene.frame_set(frame_idx)

                bpy_camera.matrix_world = world_to_blender(Matrix(self.cam_poses[frame_idx]))
                bpy_camera.keyframe_insert('location', frame=frame_idx)
                bpy_camera.keyframe_insert('rotation_euler', frame=frame_idx)

                # set a point light to move together with the camera
                lamp.matrix_world = bpy_camera.matrix_world
                lamp.keyframe_insert('location', frame=frame_idx)
                lamp.keyframe_insert('rotation_euler', frame=frame_idx)

                bpy_scene.update()

    def set_environment_lighting(self):
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)

        # set environment lighting
        light_settings = bpy.context.scene.world.light_settings
        light_settings.use_environment_light = True
        light_settings.environment_energy = np.random.uniform(self.params['env_light_min'],
            self.params['env_light_max'])
        light_settings.environment_color = 'PLAIN'

    def log_message(self, message):
        elapsed_time = time.time() - self.start_time
        print("[%.2f s] %s" % (elapsed_time, message))

if __name__ == '__main__':

    shapenet_path = '/is/sg/zlv/data-avg/datasets/ShapeNetCore.v1'

    import argparse
    parser = argparse.ArgumentParser(description='Generate 3D shapes')
    parser.add_argument('--shape_id', type=str, default='None',
        help='the shapenet id')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    shape_net_render = ShapeNetRender(shapenet_path)

    shape_net_render.run_rendering()
