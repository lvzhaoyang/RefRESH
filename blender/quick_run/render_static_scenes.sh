#!/bin/bash

# set your blender path
BLENDER_PATH=~/develop/blender-2.79b/

stride_list="1 2 5 10"
scene_list="apt0 apt1 apt2 copyroom office0 office1 office2 office3"

JOB=${1:-'--dataset bundlefusion --scene apt1 --stride 1'}

### RUN blender background rendering
$BLENDER_PATH/blender --background -P render_static_scenes.py -- ${JOB}

### Parse the final output from the OpenEXR format generated from blender
python parse_static_scene.py -- ${JOB}
