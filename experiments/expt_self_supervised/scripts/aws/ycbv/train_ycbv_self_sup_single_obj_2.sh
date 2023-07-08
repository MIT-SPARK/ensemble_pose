#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../../ || exit
#python run_training.py ycbv --model_id=obj_000001 --config="$1"
#python run_training.py ycbv --model_id=obj_000002 --config="$1"
#python run_training.py ycbv --model_id=obj_000003 --config="$1"
#python run_training.py ycbv --model_id=obj_000004 --config="$1"
python run_training.py ycbv --model_id=obj_000005 --config="$1"
python run_training.py ycbv --model_id=obj_000006 --config="$1"
python run_training.py ycbv --model_id=obj_000007 --config="$1"
#python run_training.py ycbv --model_id=obj_000008 --config="$1"
#python run_training.py ycbv --model_id=obj_000009 --config="$1"
#python run_training.py ycbv --model_id=obj_000010 --config="$1"
#python run_training.py ycbv --model_id=obj_000011 --config="$1"
#python run_training.py ycbv --model_id=obj_000012 --config="$1"
#python run_training.py ycbv --model_id=obj_000013 --config="$1"
#python run_training.py ycbv --model_id=obj_000014 --config="$1"
#python run_training.py ycbv --model_id=obj_000015 --config="$1"
#python run_training.py ycbv --model_id=obj_000016 --config="$1"
#python run_training.py ycbv --model_id=obj_000017 --config="$1"
#python run_training.py ycbv --model_id=obj_000018 --config="$1"
#python run_training.py ycbv --model_id=obj_000019 --config="$1"
#python run_training.py ycbv --model_id=obj_000020 --config="$1"
#python run_training.py ycbv --model_id=obj_000021 --config="$1"
#tar -czvf exp_synth_supervised_single_obj_ycbv_models_$TIMESTAMP.tar.gz ./exp_results/
