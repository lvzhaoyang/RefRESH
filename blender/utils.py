from mathutils import Matrix, Vector, Quaternion, Euler

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
    return T * origin

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
