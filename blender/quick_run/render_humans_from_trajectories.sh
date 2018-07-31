#!/bin/bash

# set your blender path
BLENDER_PATH=~/develop/blender-2.79b/

# Set parameters to render a diverse set of foreground
# fg_density: add a foreground object for every N frame (visible from the camera view)
# fg_stride: the motion of foreground as fg_stride*bg_stride
# bg_scene: the background scene trajetory
# bg_stride: the background stride w.r.t. the generate scene

JOB_PARAMS=${1:-'--fg_density 10 --fg_stride 20 --bg_scene apt0 --bg_stride 1 --bg_start 0 --bg_end 100'}

### RUN blender background rendering
$BLENDER_PATH/blender --background -P render_humans_from_trajectories.py -- ${JOB_PARAMS}

### Parse the final output from the OpenEXR format generated from blender
python parse_humans_from_trajectories.py -- ${JOB_PARAMS}
