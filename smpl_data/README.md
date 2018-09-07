# Data Folder for SMPL Files

This folder should finally contain:

``` 
---- textures # the texture folder made from from SURREAL dataset
---- basicModel_f_lbs_10_207_0_v1.0.0.pkl  
---- basicModel_f_lbs_10_207_0_v1.0.2.fbx  
---- basicModel_m_lbs_10_207_0_v1.0.0.pkl  
---- basicModel_m_lbs_10_207_0_v1.0.2.fbx
---- male_beta_stds.npy
---- female_beta_stds.npy
---- smpl_data.npz
```

Lacking any of the files will result in an error during dataset generation. Then place to refer the following steps to fix the dataset links. 

If there is any file missing, please doule check the following steps. 

## Download Data

### Download SMPL for MAYA

You need to download SMPL for MAYA from the [official website(click here)][3] in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files:

```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files in the current directory.

### Download SMPL textures and other relevant data

With the same credentials as with the SURREAL dataset, with the `smpl_data` folder, you can download the remaining necessary SMPL data. All the downloaded files should be placed within the same directory:

``` shell
cd smpl_data
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

## Create symbolic links for data

If you have directly downloaded the data within this directory, you can omit this step. I personally will like to download all the data in a shared scratch space, and then create symbolic links for all the data during usage. Suppose you also download all the above data into a directory '$SMPL_DATA', e.g '$SMPL_DATA=~/data/smpl', then you need to create symbolic links for all the needed files:

```
ln -s ~/data/smpl/smpl_data.npz
ln -s ~/data/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl
ln -s ~/data/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl  
ln -s ~/data/smpl/male_beta_stds.npy
ln -s ~/data/smpl/female_beta_stds.npy
```

[3]: http://smpl.is.tue.mpg.de


# README from SMPL

License:
--------
Copyright 2015 Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license


Getting Started:
----------------
To learn about SMPL, please visit our website: http://smpl.is.tue.mpg
You can find the SMPL paper at: http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

Visit our downloads page to download some sample FBX files for animation:
http://smpl.is.tue.mpg/downloads

The sample FBX files can be imported into most animation software, like Maya, Blender or 
Unreal Engine, and played. You can also find plugins and scripts for animating and reshaping your own models on the downloads page. Currently we only provide a plugin for Maya, but more scripts for other animation programs are coming soon! 

For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
-----------
This is a demo version of the script for driving the SMPL model inside Maya.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


Current versions supported:
--------------------------
Mac OSX: Maya 2013+
Windows: Maya 2014+


Dependencies:
------------
Numpy is required for running this script. Numpy is a python module that
can be installed following the instructions given here:
http://docs.scipy.org/doc/numpy/user/install.html

or here:
http://blog.animateshmanimate.com/post/115538511578/python-numpy-and-maya-osx-and-windows

Please make sure you have numpy installed on your computer and accessible through Maya's python.
We are working towards removing this dependency. 



About the Script:
-----------------
The script displays a UI to apply SMPL's shape and pose blendshapes and to adjust the skeleton to new body shapes.
Load this plugin into Maya. It will create a window with 3 options:

1- Apply Pose Blend Shapes to Current Frame: 
	If you repose the model in Maya, then click this to 
	compute and apply the pose blend shapes in the current frame. 
    You can als ochoose whether or not to set the keyframes for the 
    pose blendshapes. Check the 'Reset Keyframes' checkbox if you 
    would like to lock blendShape values at given frame by setting 
    a keyframe. 

2- Apply Pose Blend Shapes to Frames in above Range: 
	Specify a range of frames in an animation and then compute/apply 
	the pose blendshapes for all the frames in range. Check the 
    'Reset Keyframes' checkbox if you would like to lock blendShape 
    values at given frame range by setting a keyframe at each frame in the 
    given range.
    
3- Set Mesh to Bind-Pose & Recompute Skeleton: 
	When you edit the shape blend shapes to change body shape the 
	skeleton will no longer be correct.  Click first button to set the 
    mesh into the bind-pose. Next, click this to 'Recompute Skeleton' 
    to recompute the skeleton rig to match the new body shape.

Always make sure to cilck on the mesh in the 3D view to select it before 
using any of the functions in the plugin. Select only the mesh of the model 
you want to update and then click the appropriate button on the UI.