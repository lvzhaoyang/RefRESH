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

import bpy
import math
import camera
import random

from mathutils import Matrix, Vector

def initialize_objects():
    # remove the initialize cube
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

def create_blender_keyframes(poses, depths, K, object_names):
    '''

    poses:  the camera poses pre-calculated from ground truth
    depths: the real depth pre-recorded by the 3D active sensor
    K: camera intrinsics, pre-set by the 3D static scene
    object_names: the loaded synthetic objects
    '''

    bl_camera = bpy.data.objects['Camera']
    bl_scene = bpy.data.scenes['Scene']
    bl_lamp = bpy.data.objects['Lamp']

    bl_scene.render.resolution_x = 640
    bl_scene.render.resolution_y = 480

    # set the camera intrinsics
    camera.set_intrinsic(K, bl_camera, bl_scene)

    # set the environment initial settings
    bl_camera.matrix_world = world_to_blender(Matrix(poses[0]))
    bl_lamp.location = bl_camera.location

    bl_objects, bl_trajs = [], []

    for object_name in object_names:
        bl_object, bl_traj = create_flying_object(bpy.data.objects[object_name],
            bl_camera, bl_scene, K, depths, poses)
        bl_objects.append(bl_object)
        bl_trajs.append(bl_traj)

        #ensure origin is centered on bounding box center
        #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        #create a cube for the bounding box
        #bpy.ops.mesh.primitive_cube_add()
        # #our new cube is now the active object, so we can keep track of it in a variable:
        # bound_box = bl_object
        #
        # #copy transforms
        # bound_box.dimensions = obj.dimensions
        # bound_box.location = obj.location
        # bound_box.rotation_euler = obj.rotation_euler

    # define a trajectory for each of the objects

    bpy.context.scene.update()

    bl_scene.frame_current = 0
    total_frames = len(poses)
    for index in range(total_frames):
        pose = poses[index]
        bl_camera.matrix_world = world_to_blender(Matrix(pose))

        # update the scene about the camera matrices
        bpy.context.scene.update()

        # add the key frames to the scene
        bl_camera.keyframe_insert(data_path='location', frame=index,  group='LocRot')
        bl_camera.keyframe_insert(data_path='rotation_euler', frame=index, group='LocRot')
        bl_camera.keyframe_insert(data_path='scale',frame=index, group='LocRot')

        bl_lamp.location = bl_camera.location
