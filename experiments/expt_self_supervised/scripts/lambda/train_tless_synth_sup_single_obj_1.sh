#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../
python run_training.py tless --model_id=obj_000016 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000017 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000018 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000019 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000020 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000021 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000022 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000023 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000024 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000025 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000026 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000027 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000028 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000029 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
python run_training.py tless --model_id=obj_000030 --config=./configs/synth_supervised_single_obj/tless/depth_rgb_nopretrain.yml
