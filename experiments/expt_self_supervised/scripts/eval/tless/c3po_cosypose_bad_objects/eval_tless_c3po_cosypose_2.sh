#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../../../ || exit
#python run_training.py tless --model_id=obj_000001 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000001/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000002 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000002/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000003 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000003/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000004 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000004/_best_model.pth.tar
python run_training.py tless --model_id=obj_000005 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000005/_best_model.pth.tar
python run_training.py tless --model_id=obj_000006 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000006/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000007 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000007/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000008 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000008/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000009 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000009/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000010 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000010/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000011 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000011/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000012 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000012/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000013 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000013/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000014 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000014/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000015 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000015/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000016 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000016/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000017 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000017/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000018 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000018/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000019 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000019/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000020 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000020/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000021 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000021/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000022 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000022/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000023 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000023/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000024 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000024/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000025 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000025/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000026 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000026/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000027 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000027/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000028 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000028/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000029 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000029/_best_model.pth.tar
#python run_training.py tless --model_id=obj_000030 --config=./configs/eval/tless/c3po_cosypose_2_refine.yml --checkpoint_path=./exp_results/self_sup_models_best/tless/obj_000030/_best_model.pth.tar
#tar -czvf exp_synth_supervised_single_obj_tless_models_$TIMESTAMP.tar.gz ./exp_results/
