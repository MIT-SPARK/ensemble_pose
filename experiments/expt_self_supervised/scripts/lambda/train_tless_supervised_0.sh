#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd ../
python run_supervised_training.py tless --model_id=obj_000001 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000002 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000003 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000004 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000005 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000006 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000007 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000008 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000009 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000010 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000011 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000012 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000013 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000014 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000015 --config=./configs/supervised_tless.yml
