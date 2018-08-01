#!/bin/bash

download_scene() {
    # download the zipped color and depth images, with the camera parameters
    if [ -d "$1/$2" ]; then
        echo "Directory $1/$2 exists. Move to the next..."
    else
        if [ -f "$1/$2.zip" ]; then
            echo "File $1/$2.zip exists. Unzip it..."
            unzip "$1/$2.zip" -d "$1"
        else
            echo "File $1/$2.zip does not exist. Download and unzip it ..."
            wget $3 -P "$dest"
            unzip "$1/$2.zip" -d "$1"          
        fi
    fi
    # download the point cloud data
    if [ -f "$1/$2.ply" ]; then
        echo "Point clout $1/$2.ply exists. Move to the next..."
    else
        echo "Point clout $1/$2.ply does not exists. Download it..."
        wget $4 -P $dest
    fi
}

# choose a destination of the data folder
dest="data/RefRESH/BundleFusion/raw"

# download all the scenes
download_scene "$dest" "apt0" http://graphics.stanford.edu/projects/bundlefusion/data/apt0/apt0.zip http://graphics.stanford.edu/projects/bundlefusion/data/apt0/apt0.ply
download_scene "$dest" "apt1" http://graphics.stanford.edu/projects/bundlefusion/data/apt1/apt1.zip http://graphics.stanford.edu/projects/bundlefusion/data/apt1/apt1.ply 
download_scene "$dest" "apt2" http://graphics.stanford.edu/projects/bundlefusion/data/apt2/apt2.zip http://graphics.stanford.edu/projects/bundlefusion/data/apt2/apt2.ply 
download_scene "$dest" "copyroom" http://graphics.stanford.edu/projects/bundlefusion/data/copyroom/copyroom.zip http://graphics.stanford.edu/projects/bundlefusion/data/copyroom/copyroom.ply 
download_scene "$dest" "office0" http://graphics.stanford.edu/projects/bundlefusion/data/office0/office0.zip http://graphics.stanford.edu/projects/bundlefusion/data/office0/office0.ply 
download_scene "$dest" "office1" http://graphics.stanford.edu/projects/bundlefusion/data/office1/office1.zip http://graphics.stanford.edu/projects/bundlefusion/data/office1/office1.ply 
download_scene "$dest" "office2" http://graphics.stanford.edu/projects/bundlefusion/data/office2/office2.zip http://graphics.stanford.edu/projects/bundlefusion/data/office2/office2.ply 
download_scene "$dest" "office3" http://graphics.stanford.edu/projects/bundlefusion/data/office3/office3.zip http://graphics.stanford.edu/projects/bundlefusion/data/office3/office3.ply 