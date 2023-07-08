#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd ../
python run_supervised_training.py ycbv --model_id=obj_000011 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000012 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000013 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000014 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000015 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000016 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000017 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000018 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000019 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000020 --config=./configs/supervised_ycbv.yml
python run_supervised_training.py ycbv --model_id=obj_000021 --config=./configs/supervised_ycbv.yml
