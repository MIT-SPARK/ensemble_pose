#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd ../
#python run_training.py ycbv --model_id=obj_000001 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000002 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000003 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000004 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000005 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000006 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000007 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000008 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000009 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#python run_training.py ycbv --model_id=obj_000010 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000011 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000012 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000013 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000014 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000015 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000016 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000017 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000018 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000019 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000020 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
python run_training.py ycbv --model_id=obj_000021 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
#tar -czvf exp_synth_supervised_single_obj_ycbv_models_$TIMESTAMP.tar.gz ./exp_results/
