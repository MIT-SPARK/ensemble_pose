#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd $SCRIPT_DIR/../../
#python run_training.py ycbv --model_id=$1 --config=./configs/ablations/ycbv/proposed_ds_50.yml
#python run_training.py ycbv --model_id=$1 --config=./configs/ablations/ycbv/pooling_rand_ds_50.yml
python run_training.py ycbv --model_id=$1 --config=./configs/ablations/ycbv/pooling_fps_ds_50.yml
