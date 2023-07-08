docker run -it --user="$(id -u $USER)":"$(id -g $USER)" \
--gpus '"device=1"' \
--shm-size 16G \
--env="DISPLAY=:1" \
--env="CUDA_VISIBLE_DEVICES=0" \
--env="PYTHONPATH=$PYTHONPATH:/opt/project/ensemble_pose/src:/opt/project/ensemble_pose/external/bop_toolkit:/opt/project/ensemble_pose/external:/opt/project/ensemble_pose/experiments" \
--volume="/home/$(whoami)/code/ensemble_pose/:/opt/project/ensemble_pose" \
--volume="/exp_tmp/datasets/:/mnt/datasets/" \
casper/pytorch1.12.1:latest bash
