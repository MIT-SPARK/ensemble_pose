#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../../ || exit
python run_training.py ycbv --model_id=obj_000001 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000001/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000002 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000002/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000003 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000003/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000004 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000004/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000005 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000005/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000006 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000006/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000007 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000007/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000008 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000008/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000009 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000009/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000010 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000010/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000011 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000011/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000012 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000012/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000013 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000013/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000014 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000014/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000015 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000015/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000016 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000016/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000017 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000017/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000018 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000018/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000019 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000019/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000020 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000020/_best_model.pth.tar
python run_training.py ycbv --model_id=obj_000021 --config=./configs/eval/ycbv/c3po_cosypose_2_refine_bop_dets_retrained.yml --checkpoint_path=./exp_results/self_sup_models_best_bop_dets/ycbv/obj_000021/_best_model.pth.tar
#tar -czvf exp_synth_supervised_single_obj_ycbv_models_$TIMESTAMP.tar.gz ./exp_results/
