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

=============================================================================

This file refactored on top of SURREAL: https://github.com/gulvarol/surreal

If you use this file, please check their license

https://github.com/gulvarol/surreal/blob/master/LICENSE.md
"""

import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam
from mathutils import Matrix, Vector, Quaternion, Euler
from pickle import load
from random import choice, random, seed

from utils import world_to_blender, blender_to_world

# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

# smpl body part information
sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']

class SMPL_Body:
    """ The class for SMPL human body
    """
    armature = None
    armature_name = None
    obj_name = None
    obj = None
    gender = None

    body_data = None

    orig_pelvis_loc = None

    m_fbx = '../smpl_data/basicModel_m_lbs_10_207_0_v1.0.2.fbx' # male MAYA file
    f_fbx = '../smpl_data/basicModel_f_lbs_10_207_0_v1.0.2.fbx' # female MAYA file

    def __init__(self, smpl_data, gender, cam_pose, material, idx=0, anchor_location3d=None):

        if gender == 'male':
            fbx_file_path = self.m_fbx
        elif gender == 'female':
            fbx_file_path = self.f_fbx
        else:
            NotImplementedError

        bpy.ops.import_scene.fbx(filepath=fbx_file_path, axis_forward='Y', axis_up='Z', global_scale=25)

        # all the moving human body named as Lewis_0, Lewis_1, etc.
        self.armature_name = 'Lewis_{:d}'.format(idx)
        bpy.context.active_object.name = self.armature_name
        self.armature = bpy.data.objects[self.armature_name]

        self.obj_name = 'body_{:d}'.format(idx)
        bpy.data.objects[self.armature_name].children[0].name = self.obj_name
        self.obj =  bpy.data.objects[self.obj_name]

        self.obj.data.use_auto_smooth = False

        self.gender_name = '%s_avg' % gender[0]

        seed()

        if anchor_location3d is None:
            #set a heuristic to set the object bodies
            t = Vector(np.array([-0.86, -0.6, -0.5]) + np.random.normal(0, 0.1, 3))
        else:
            t = Vector(anchor_location3d)

        # random facing the camera
        r = np.array([-np.pi/2, -np.pi/2, 0]) + np.random.normal(0, 0.2, 3)

        self.armature.location = t
        self.armature.rotation_euler = Euler(r, 'XYZ')
        bpy.context.scene.update()

        self.armature.matrix_world = cam_pose * self.armature.matrix_world

        self.armature.location -= Vector((0.25, -0.15, 0.0))
        bpy.context.scene.update()


        self.armature.animation_data_clear()

        # create material segmentation
        self.obj.select = True
        bpy.context.scene.objects.active = self.obj
        segmented_materials = True  #True: 0-24, False: expected to have 0-1 bg/fg

        # create material segmentation
        if segmented_materials:
            self.materials = create_body_segmentation(self.obj, material)
            prob_dressed = {'leftLeg':.5, 'leftArm':.9, 'leftHandIndex1':.01,
                            'rightShoulder':.8, 'rightHand':.01, 'neck':.01,
                            'rightToeBase':.9, 'leftShoulder':.8, 'leftToeBase':.9,
                            'rightForeArm':.5, 'leftHand':.01, 'spine':.9,
                            'leftFoot':.9, 'leftUpLeg':.9, 'rightUpLeg':.9,
                            'rightFoot':.9, 'head':.01, 'leftForeArm':.5,
                            'rightArm':.5, 'spine1':.9, 'hips':.9,
                            'rightHandIndex1':.01, 'spine2':.9, 'rightLeg':.5}
        else:
            self.materials = {'FullBody': material}
            prob_dressed = {'FullBody': .6}

        self.orig_pelvis_loc = (self.armature.matrix_world.copy() * self.armature.pose.bones[self.gender_name+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
        # self.orig_pelvis_loc = self.armature.location.copy() + Vector((-2., -1.5, 1.))

        # unblocking both the pose and the blendshape limits
        for k in self.obj.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

        # load body data
        self.body_data, fshapes, name = load_body_data(smpl_data, self.obj, self.gender_name, gender=gender)

        print("Loaded body data for %s" % name)

        # pick random real body shape
        self.shape = choice(fshapes) #+random_shape(.5) can add noise

        bpy.context.scene.objects.active = self.armature
        orig_trans = np.asarray(self.armature.pose.bones[self.gender_name+'_Pelvis'].location).copy()

        self.armature.animation_data_clear()

    # apply trans pose and shape to character
    def apply_Rt_body_shape(self, body_data_idx, frame=None):
        '''
        Input:
        @param body_data_index
        @param frame
        '''

        body_data_len = len(self.body_data['poses'])
        body_data_idx = body_data_idx % body_data_len

        pose  = self.body_data['poses'][body_data_idx]
        location = self.body_data['trans'][body_data_idx]
        trans = Vector(location)

        # transform pose into rotation matrices (for pose) and pose blendshapes
        mrots, bsh = rodrigues2bshapes(pose)

        # set the location of the first bone to the translation parameter
        self.armature.pose.bones[self.gender_name+'_Pelvis'].location = trans
        if frame is not None:
            self.armature.pose.bones[self.gender_name+'_root'].keyframe_insert('location', frame=frame)
        # set the pose of each bone to the quaternion specified by pose
        for bone_idx, mrot in enumerate(mrots):
            bone = self.armature.pose.bones[self.gender_name+'_'+part_match['bone_%02d' % bone_idx]]
            bone.rotation_quaternion = Matrix(mrot).to_quaternion()
            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)
                bone.keyframe_insert('location', frame=frame)

        # apply pose blendshapes
        for ibshape, bshape in enumerate(bsh):
            self.obj.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
            if frame is not None:
                self.obj.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

        # apply shape blendshapes
        for ibshape, shape_elem in enumerate(self.shape):
            self.obj.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
            if frame is not None:
                self.obj.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

        return pose, location

    def get_bone_locs(self, scene, bpy_camera_obj):
        '''
        Calculate the bone location, given the camera parameters
        '''

        n_bones = 24
        render_scale = scene.render.resolution_percentage / 100
        render_size = (int(scene.render.resolution_x * render_scale),
                       int(scene.render.resolution_y * render_scale))
        bone_locations_2d = np.empty((n_bones, 2))
        bone_locations_3d = np.empty((n_bones, 3), dtype='float32')

        # obtain the coordinates of each bone head in image space
        for bone_idx in range(n_bones):
            bone = self.armature.pose.bones[self.gender_name+'_'+part_match['bone_%02d' % bone_idx]]
            co_3d = self.armature.matrix_world * bone.head
            co_2d = world2cam(scene, bpy_camera_obj, co_3d)
            bone_locations_3d[bone_idx] = (co_3d.x,
                                     co_3d.y,
                                     co_3d.z)
            bone_locations_2d[bone_idx] = (round(co_2d.x * render_size[0]),
                                     round(co_2d.y * render_size[1]))
        return(bone_locations_2d, bone_locations_3d)


    def reset_pose(self):
        self.armature.pose.bones[self.gender_name+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))


# load poses and shapes
def load_body_data(smpl_data, ob, obj_name, gender='female', idx=None):
    '''
    load MoSHed data from CMU Mocap (only the given idx is loaded)

    example shapes:
        shape = np.zeros(10) #average

        #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #FAT

        #shape = np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]) #THIN

        #shape = np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]) #SHORT

        #shape = np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]) #TALL

    '''

    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    if idx is None:
        idx = int(random() * 1e4)

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms[name], fshapes, name)

# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
# TODO: we may do not need this since we don't need segmentation in material pass
def create_body_segmentation(ob, material):
    '''
    material is the input material map for different objects
    '''

    part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

    materials = {}
    vgroups = {}
    with open('../pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = material.copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    '''
    Return:
    @param mat_rots: Rodrigues representation for rotations
    @param bshapes: blender shapes
    '''

    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obj_name, scene, bpy_camera_obj,
    reg_ivs, joint_reg):
    '''
    since the regression is sparse, only the relevant vertex elements (joint_reg)
    and their indices (reg_ivs) are loaded
    '''
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_Rt_body_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obj_name, scene, bpy_camera_obj)

    bpy.ops.wm.memory_statistics() # print memory statistics

    # obtain a mesh after applying modifiers
    # obj_mesh holds the vertices after applying the shape blendshapes
    obj_mesh = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = obj_mesh.vertices[iv].co
    bpy.data.meshes.remove(obj_mesh)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for bone_idx in range(24):
        bb = arm_ob.data.edit_bones[obj_name + '_' + part_match['bone_%02d' % bone_idx]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[bone_idx]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)
