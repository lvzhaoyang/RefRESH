#!/bin/bash

# set your blender path
BLENDER_PATH=~/develop/blender-2.79b/

FG_STRIDE=10 # the average stride of motions sampled from the MoCap
FG_NUM=10 # number of synthetic humans per 100 frames

JOB_PARAMS=${1:-'--fg_stride ${FG_STRIDE} --fg_number ${FG_NUM}'}

### RUN blender background rendering
$BLENDER_PATH/blender -P render_bodies.py -- ${JOB_PARAMS}

### Parse the output for all scenes in bundlefusion
#python parse_static_scene_output.py -- ${JOB}
