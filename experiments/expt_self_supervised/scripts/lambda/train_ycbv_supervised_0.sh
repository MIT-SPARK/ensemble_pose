#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd ../
python run_supervised_training.py ycbv --model_id=obj_000001 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000002 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000003 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000004 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000005 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000006 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000007 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000008 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000009 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000010 --config=./configs/supervised_ycbv.yml