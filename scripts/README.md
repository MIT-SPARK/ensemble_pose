# Scripts
This folder contains useful scripts that we use for this project.

## `gen_sdf_from_meshes.py`
This is a command-line tool for generating SDFs from meshes.

Usage:
```bash
python gen_sdf_from_meshes.py --method mesh2sdf --config /home/jnshi/code/CASPER-3D/scripts/configs/sdf_gen.yml
--input_dir /mnt/jnshi_data/datasets/casper-data/data/sdf_libs/meshes/chair_test
--output_dir /mnt/jnshi_data/datasets/casper-data/data/sdf_libs/sdfs/chair_test
--verbose
```
If you are inside the provided Docker image, run:
```bash
PYTHONPATH="${PYTHONPATH}:/opt/project/CASPER-3D/src" python /opt/project/CASPER-3D/scripts/gen_sdf_from_meshes.py \
--method mesh2sdf_floodfill \
--config /opt/project/CASPER-3D/scripts/configs/sdf_gen.yml \
--input_dir /mnt/datasets/sdf_lib/test_meshes/ \
--output_sdf_dir /mnt/datasets/sdf_lib/test_sdf/ \
--output_grad_dir /mnt/datasets/sdf_lib/test_sdf_grad/ -d 
```