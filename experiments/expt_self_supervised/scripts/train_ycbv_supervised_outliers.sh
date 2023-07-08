#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cd ../
python run_supervised_training.py ycbv --model_id=obj_000001 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000002 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000003 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000004 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000005 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000006 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000007 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000008 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000009 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000010 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000011 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000012 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000013 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000014 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000015 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000016 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000017 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000018 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000019 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000020 --config=./configs/supervised_ycbv_outliers.yml
python run_supervised_training.py ycbv --model_id=obj_000021 --config=./configs/supervised_ycbv_outliers.yml
tar -czvf exp_supervised_outliers_ycbv_models_$TIMESTAMP.tar.gz ./exp_results/supervised_outliers/ycbv
