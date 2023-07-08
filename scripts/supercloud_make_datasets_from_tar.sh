#!/bin/bash
echo "Received arguments for datasets generation: $1 $2"

BOP_TARS_DIR="/home/gridsan/jshi/bop_datasets_tars/"
BOP_IMAGES_ZIP_DIR="/home/gridsan/jshi/bop_datasets_compressed/"
DEST_DATA_DIR="$TMPDIR/data/"

mkdir -p $DEST_DATA_DIR

# extract the bop data folder
tar -xvf $BOP_TARS_DIR/bop_annotated_base.tar -C $DEST_DATA_DIR

if [ $1 = "ycbv" ]; then
  # extract ycbv annotated base
  tar -xvf $BOP_TARS_DIR/ycbv_annotated_base.tar -C $DEST_DATA_DIR

  # extract ycbv image data
  if [ $2 = "test" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_test_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
  fi
  if [ $2 = "train_pbr" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
  fi
  if [ $2 = "train_real" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_real.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
  fi
  if [ $2 = "train_synt" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_train_synt.zip -d $DEST_DATA_DIR/bop/bop_datasets/ycbv
  fi
fi

if [ $1 = "tless" ]; then
  # extract tless annotated base
  tar -xvf $BOP_TARS_DIR/tless_annotated_base.tar -C $DEST_DATA_DIR

  # extract tless image data
  if [ $2 = "test" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/ycbv_test_primesense_all.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
  fi
  if [ $2 = "train_pbr" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_pbr.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
  fi
  if [ $2 = "train_primesense" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_primesense.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
  fi
  if [ $2 = "train_render_reconst" ]; then
    unzip -q $BOP_IMAGES_ZIP_DIR/tless_train_render_reconst.zip -d $DEST_DATA_DIR/bop/bop_datasets/tless
  fi
fi
