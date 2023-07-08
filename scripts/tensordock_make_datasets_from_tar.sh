#!/bin/bash
echo "Received arguments for datasets generation: $1 $2"

BOP_TARS_DIR="/home/$(whoami)/datasets/bop_datasets_tars/"
BOP_IMAGES_ZIP_DIR="/home/$(whoami)/datasets/bop_datasets_compressed/"
DEST_DATA_DIR="/home/$(whoami)/datasets/"

mkdir -p $DEST_DATA_DIR

# extract the bop data folder
tar -xvf $BOP_TARS_DIR/bop_annotated_base.tar -C $DEST_DATA_DIR

if [ $1 = "ycbv" ]; then
  # extract ycbv annotated base
  tar -xvf $BOP_TARS_DIR/ycbv_annotated_base.tar -C $DEST_DATA_DIR

  # extract ycbv image data
  unzip -o $BOP_IMAGES_ZIP_DIR/ycbv_test_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/ycbv_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/ycbv_train_real.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/ycbv_train_synt.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv | awk 'BEGIN {ORS=" "} {print "."}'
fi

if [ $1 = "tless" ]; then
  # extract tless annotated base
  tar -xvf $BOP_TARS_DIR/tless_annotated_base.tar -C $DEST_DATA_DIR

  # extract tless image data
  unzip -o $BOP_IMAGES_ZIP_DIR/tless_test_primesense_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/tless_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/tless_train_primesense.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless | awk 'BEGIN {ORS=" "} {print "."}'
  unzip -o $BOP_IMAGES_ZIP_DIR/tless_train_render_reconst.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless | awk 'BEGIN {ORS=" "} {print "."}'
fi
