#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../

python eval/eval_c3po.py ycbv.test "$1" ./exp_results/ablations/proposed/ycbv/"$1"/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_proposed --config=./exp_results/ablations/proposed/ycbv/"$1"/config.yml
python eval/eval_c3po.py ycbv.test "$1" ./exp_results/ablations/fps/ycbv/"$1"/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_fps --config=./exp_results/ablations/fps/ycbv/"$1"/config.yml
python eval/eval_c3po.py ycbv.test "$1" ./exp_results/ablations/rand_sampling/ycbv/"$1"/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar ablations_rand_sampling --config=./exp_results/ablations/rand_sampling/ycbv/"$1"/config.yml
