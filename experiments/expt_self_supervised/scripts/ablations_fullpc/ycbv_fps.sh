#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
python run_training.py ycbv --model_id=obj_000001 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000002 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000003 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000004 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000005 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000006 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000007 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000008 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000009 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000010 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000011 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000012 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000013 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000014 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000015 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000016 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000017 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000018 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000019 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000020 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
python run_training.py ycbv --model_id=obj_000021 --config=./configs/fullpc_supervised/ycbv/depth_fps.yml
#tar -czvf exp_supervised_ycbv_models_$TIMESTAMP.tar.gz ./exp_results/supervised/ycbv
