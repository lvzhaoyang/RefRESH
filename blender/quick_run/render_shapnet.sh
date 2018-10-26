#!/bin/bash

# set your blender path
BLENDER_PATH=~/develop/blender-2.79b/

#NUM_OBJ = 10 # the average number of objects sampled from shapenet

JOB=${1:-'--seq_num 200 --start_index 100 --object_id 02858304'}

# $BLENDER_PATH/blender -P render_shapenet.py -- ${JOB}
$BLENDER_PATH/blender --background -P render_shapenet.py -- ${JOB}

# parse the final output from the OpenEXR format generated from blender
python parse_shapenet.py -- ${JOB}
