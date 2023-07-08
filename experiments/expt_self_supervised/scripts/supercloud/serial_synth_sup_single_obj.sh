#!/bin/bash

# variables
HOME_DIR=/home/gridsan/jshi
REPO_DIR=$HOME_DIR/CASPER-3D

# extract datasets
cd $REPO_DIR/scripts || exit
bash supercloud_make_datasets_from_tar.sh ycbv train_pbr

# set up symlink
cd $REPO_DIR || exit
ln -s $TMPDIR/data ./data

# setup python path
export PYTHONPATH=$PYTHONPATH:$REPO_DIR/src:$REPO_DIR/external/bop_toolkit:$REPO_DIR/external:$REPO_DIR/experiments
export CUDA_VISIBLE_DEVICES=0