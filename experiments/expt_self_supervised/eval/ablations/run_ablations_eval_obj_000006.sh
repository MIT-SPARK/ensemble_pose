#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../

python eval/eval_c3po.py ycbv.pbr obj_000006 ./exp_results/ablations/proposed/ycbv/obj_000006/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_proposed --config=./exp_results/ablations/proposed/ycbv/obj_000018/config.yml --indices=./eval/ablations/ablation_val_indices_obj_000006.npy
python eval/eval_c3po.py ycbv.pbr obj_000006 ./exp_results/ablations/fps/ycbv/obj_000006/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_fps --config=./exp_results/ablations/fps/ycbv/obj_000018/config.yml --indices=./eval/ablations/ablation_val_indices_obj_000006.npy
python eval/eval_c3po.py ycbv.pbr obj_000006 ./exp_results/ablations/rand_sampling/ycbv/obj_000006/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_rand_sampling --config=./exp_results/ablations/rand_sampling/ycbv/obj_000018/config.yml --indices=./eval/ablations/ablation_val_indices_obj_000006.npy
python eval/eval_c3po.py ycbv.pbr obj_000006 ./exp_results/ablations/nonrobust_centroid/ycbv/obj_000006/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_nonrobust_centroid --config=./exp_results/ablations/nonrobust_centroid/ycbv/obj_000018/config.yml --indices=./eval/ablations/ablation_val_indices_obj_000006.npy
