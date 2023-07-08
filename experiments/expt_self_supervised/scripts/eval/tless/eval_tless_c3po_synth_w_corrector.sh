#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../../ || exit
python run_training.py tless --model_id=obj_000001 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000002 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000003 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000004 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000005 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000006 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000007 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000008 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000009 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000010 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000011 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000012 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000013 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000014 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000015 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000016 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000017 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000018 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000019 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000020 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000021 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000022 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000023 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000024 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000025 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000026 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000027 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000028 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000029 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
python run_training.py tless --model_id=obj_000030 --config=./configs/eval/tless/c3po_synth_w_corrector.yml
#tar -czvf exp_synth_supervised_single_obj_tless_models_$TIMESTAMP.tar.gz ./exp_results/
