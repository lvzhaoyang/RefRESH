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

from mathutils import Matrix, Vector, Quaternion, Euler
import math

def world_to_blender(T):
    origin = Matrix(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return T * origin

def blender_to_world(T):
    transform = Matrix(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return T * transform

def set_intrinsic(K, bpy_camera, bpy_scene, H, W):
    '''
    Set camera intrinsic parameters given the camera calibration. This function is written refering to https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model (an inverse version)

    @ToDo: currently it does not contain the possible principle point shift in blender.

    :param K camera intrinsic parameters
    :param bpy_camera: blender camera object
    :param bpy_scene: blender scene object
    '''

    fx, fy, cx, cy = K

    resolution_x_in_px = W
    resolution_y_in_px = H

    bpy_scene.render.resolution_x = resolution_x_in_px
    bpy_scene.render.resolution_y = resolution_y_in_px

    scale = resolution_x_in_px / (2*cx)

    bpy_scene.render.resolution_percentage = scale * 100

    bpy_scene.render.pixel_aspect_x = 1.0
    bpy_scene.render.pixel_aspect_y = 1.0

    # both in mm
    bpy_camera.data.sensor_width  = fy * cx / (fx * cy)
    bpy_camera.data.sensor_height = 1 # does not matter

    s_u = resolution_x_in_px / bpy_camera.data.sensor_width
    s_v = resolution_y_in_px / bpy_camera.data.sensor_height

    # we will use the default blender camera focal length
    bpy_camera.data.lens = fx / s_u

def parent_obj_to_camera(bpy_camera, bpy_scene, bpy_origin):
    """ Set the camera's parent to be origin
    """
    origin = (0, 0, 0)
    bpy_origin.location = origin
    bpy_camera.parent = bpy_origin  # setup parenting

    bpy_scene.objects.link(bpy_origin)
    bpy_scene.objects.active = bpy_origin
    return bpy_origin

def allocentric_pose(dist, azimuth_deg, elevation_deg):
    """ Set the allocentric (camera) pose in Polar coordinate
    """
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return Vector((x, y, z))

def ypr2quaternion(yaw, pitch, roll):
    """ Transform the (yaw, pitch, roll) representation to quaternion
    Implementation from RenderForCNN (quaternionFromYawPitchRoll):
    https://github.com/ShapeNet/RenderForCNN/blob/master/render_pipeline/render_model_views.py
    Check their license (MIT):
    https://github.com/ShapeNet/RenderForCNN/blob/master/LICENSE
    """
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return Vector((q1, q2, q3, q4))
