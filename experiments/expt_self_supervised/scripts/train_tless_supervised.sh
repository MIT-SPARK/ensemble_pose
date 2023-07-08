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
python run_supervised_training.py tless --model_id=obj_000016 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000017 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000018 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000019 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000020 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000021 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000022 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000023 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000024 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000025 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000026 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000027 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000028 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000029 --config=./configs/supervised_tless.yml
python run_supervised_training.py tless --model_id=obj_000030 --config=./configs/supervised_tless.yml
tar -czvf exp_supervised_tless_models_$TIMESTAMP.tar.gz ./exp_results/supervised/tless
