# Pipeline Overview

For the `PointsRegressionModel` used in the pipeline, all object keypoints, diameters and CAD models are in meters, 
to be consistent with the units used in CosyPose (m vs. mm).

# Training

Note that the trained keypoint detector takes in zero-centered and normalized 
(scaled so that point cloud diameter equals to 1) point clouds.
The YCBV dataloader gives the keypoint detection model normalized and zero-centered point clouds.

## With Simulated Point Clouds
In the config file, specify `train_mode` to be supervised.

Run `scripts/train_ycbv_supervised.sh` to train all objects in YCBV. Make sure to double check the `save_folder` in the 
config file.

## Evaluate Single Object Self Supervised
Evaluate:
```bash 
python run_training.py ycbv --model_id=obj_000001 --config=./configs/eval/ycbv/c3po_cosypose.yml --checkpoint_path=./exp_results/self_sup_models_best/ycbv/obj_000001/_epoch_20_self_supervised_single_obj_joint.pth.tar
```

## Single object self supervised
Self-supervised training with synthetic images on YCBV:
```bash 
python run_training.py ycbv --model_id=obj_000001 --config=./configs/self_supervised_single_obj/ycbv/depth_rgb.yml 
```

## With Synthetic Images (Single Object)
Supervised training with synthetic images on YCBV:
```bash 
python run_training.py ycbv --model_id=obj_000001 --config=./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
```

## With Synthetic Images
Supervised training with synthetic images on YCBV:
```bash 
python run_supervised_training.py ycbv --config=./configs/synth_supervised/ycbv/depth_pretrain.yml
```

Resume from checkpoint:
```bash 
python run_supervised_training.py ycbv --config=./configs/supervised_synth_ycbv.yml --resume_run \
--multimodel_checkpoint_path PATH_TO_CHECKPOINT
```

Supervised training with synthetic images on TLESS:
```bash 
python run_supervised_training.py tless --config=./configs/supervised_synth_tless.yml
```

## Self-supervised Tests (Deprecated)

Depth only on YCBV:
```bash
python run_self_supervised_training.py ycbv --config=./configs/self_supervised_ycbv_depth_only.yml
```

Depth only on TLESS:
```bash
python run_self_supervised_training.py tless --config=./configs/self_supervised_tless_depth_only.yml
```

RGB only on YCBV:
```bash 
python run_self_supervised_training.py ycbv --config=./configs/self_supervised_ycbv_rgb_only.yml
```

# Visualize Trained Models

Visualize single object C3PO model on simulated point clouds:
```bash 
python visualize_model.py ycbv obj_000001 kp_detector ./exp_results/supervised_outliers/ycbv/obj_000001/_ycbv_best_supervised_kp_point_transformer.pth
```

Visualize trained MultiModel on YCBV dataset:
```bash 
python visualize_multimodel.py ./exp_results/ycbv/20221104_230139/_epoch_50_synth_supervised_kp_point_transformer.pth.tar --config=./configs/vis_multimodel.yml
 ```

# Generating Plots and Tables
## Corrector Robustness Ablations

## Backbone Outlier Robustness Ablations

## Test Set Eval 

# Multi-Model Self-supervised Loss Logic

1. Forward pass through C3PO model
2. Forward pass through CosyPose model
3. Certify C3PO model results
4. Certify CosyPose model results

# Evaluation
## Evaluation with Simulated Data
At the experiment root directory, run
```bash 
bash eval/eval_sim_ycbv.sh
```

# Tips

If you see errors like the one below, make sure the keypoints annotations loaded match the trained models.
``` 
RuntimeError: Error(s) in loading state_dict for PointsRegressionModel:
        size mismatch for keypoint_detector.keypoint.final_block.2.weight: copying a param with shape torch.Size([21, 512]) from checkpoint, the shape in current model is torch.Size([33, 512]).
        size mismatch for keypoint_detector.keypoint.final_block.2.bias: copying a param with shape torch.Size([21]) from checkpoint, the shape in current model is torch.Size([33]).
```
This will happen if the number of keypoints in the annotations mismatch the number of keypoints in the keypoint detection model.