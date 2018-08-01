import bpy
import math
import numpy as np

from mathutils import Matrix, Vector

def set_intrinsic(K, bl_camera, bl_scene):
    '''
    Set camera intrinsic parameters given the camera calibration. This function is written refering to https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model (an inverse version)
    :param K camera intrinsic parameters
    :param bl_camera: blender camera object
    :param bl_scene: blender scene object
    '''

    fx, fy, cx, cy = K

    resolution_x_in_px = bl_scene.render.resolution_x
    resolution_y_in_px = bl_scene.render.resolution_y

    scale = resolution_x_in_px / (2*cx)

    bl_scene.render.resolution_percentage = scale * 100

    bl_scene.render.pixel_aspect_x = 1.0
    bl_scene.render.pixel_aspect_y = 1.0

    # both in mm
    bl_camera.data.sensor_width  = fy * cx / (fx * cy)
    bl_camera.data.sensor_height = 1 # does not matter

    s_u = resolution_x_in_px / bl_camera.data.sensor_width
    s_v = resolution_y_in_px / bl_camera.data.sensor_height

    # we will use the default blender camera focal length
    bl_camera.data.lens = fx / s_u

class CameraCone:
    '''
    Determine Camera frustum / cone in the scene
    '''

    def __init__(self, matrix, sensor_width, lens, resolution_x, resolution_y):
        self.matrix = matrix.inverted()
        self.sensor_width = sensor_width
        self.lens = lens

        w = 0.5* sensor_width / lens
        if resolution_x> resolution_y:
            x = w
            y = w*resolution_y/resolution_x
        else:
            x = w*resolution_x/resolution_y
            y = w

        lr = Vector([x,-y,-1])
        ur = Vector([x,y,-1])
        ll = Vector([-x,-y,-1])
        ul = Vector([-x,y,-1])
        self.half_plane_normals = [
            lr.cross(ll).normalized(),
            ll.cross(ul).normalized(),
            ul.cross(ur).normalized(),
            ur.cross(lr).normalized()
        ]

        # todo: add a bounding box for the camera and show it in the scene


    def from_camera(cam, scn):
        '''
        Calculate the camera cone given the camera and scene information
        '''
        return CameraCone(cam.matrix_world, cam.data.sensor_width, cam.data.lens, scn.render.resolution_x, scn.render.resolution_y)

    def isVisible(self, loc, fudge=0):
        '''
        Check whether a location is visible in the scene
        '''

        loc2 = self.matrix * loc

        for norm in self.half_plane_normals:
            z2 = loc2.dot(norm)
            if z2 < -fudge:
                return False

        return True
