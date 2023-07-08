#!/bin/bash

# generate the tar without annotations & object pcs
BOP_ROOT_PATH="/opt/project/CASPER-3D/data/"
OUTPUT_DIR_PATH="/opt/project/CASPER-3D/data/bop/bop_datasets_tars/"
mkdir -p $OUTPUT_DIR_PATH

cd $BOP_ROOT_PATH || exit
# bop_base: does not have bop datasets
tar -cvf $OUTPUT_DIR_PATH/bop_annotated_base.tar --exclude='*_pc_img_data.pkl' \
--exclude='bop_datasets' --exclude="bop_datasets_tars" bop

# show progress
#tar -c --exclude='*_pc_img_data.pkl' \
#       --exclude="test_primesense" \
#       --exclude="train_primesense" \
#       --exclude="train_pbr" \
#       bop_datasets/tless | pv -s $(du -sb bop_datasets/ycbv | awk '{print $1}') > $OUTPUT_DIR_PATH/tless_base.tar


# ycbv
tar -cvf $OUTPUT_DIR_PATH/ycbv_annotated_base.tar --exclude='*_pc_img_data.pkl' \
         --exclude='*_pc_data.pkl' \
         --exclude="train_pbr" \
         --exclude="train_real" \
         --exclude="train_synt" \
         --exclude="test" \
         bop/bop_datasets/ycbv

# tless
tar -cvf $OUTPUT_DIR_PATH/tless_annotated_base.tar --exclude='*_pc_img_data.pkl' \
       --exclude='*_pc_data.pkl' \
       --exclude="test_primesense" \
       --exclude="train_primesense" \
       --exclude="train_render_reconst" \
       --exclude="train_pbr" \
       bop/bop_datasets/tless
