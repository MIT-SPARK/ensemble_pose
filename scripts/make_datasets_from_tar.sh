#!/bin/bash

BOP_TARS_DIR="/opt/project/CASPER-3D/data/bop/bop_datasets_tars/"
BOP_IMAGES_ZIP_DIR="/opt/project/CASPER-3D/data/bop_datasets_compressed/"
DEST_DATA_DIR="/opt/project/CASPER-3D/test_data/"

mkdir -p DEST_DATA_DIR

# extract the bop data folder
tar -xvf $BOP_TARS_DIR/bop_annotated_base.tar -C $DEST_DATA_DIR

# extract ycbv annotated base
tar -xvf $BOP_TARS_DIR/ycbv_annotated_base.tar -C $DEST_DATA_DIR

# extract ycbv image data
unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_test_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_real.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_synt.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv

# extract tless annotated base
tar -xvf $BOP_TARS_DIR/tless_annotated_base.tar -C $DEST_DATA_DIR

# extract tless image data
unzip -q $BOP_IMAGES_ZIP_DIR/tless_test_primesense_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_primesense.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_render_reconst.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
