#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../
python run_training.py tless --model_id=obj_000001 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000002 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000003 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000004 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000005 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000006 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000007 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000008 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000009 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000010 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000011 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000012 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000013 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000014 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000015 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
