
## Parameter Tuning

This describes the process of parameter tuning for kp_corrector_reg. 
We tune epsilon and clamp_thres for each object. 
We do this by fixing added noise parameters (in tune.yml): kp_noise_fra to 0.8
and kp_noise_var_range to a list [0.5, 0.51, 1.0], which sets kp_noise_var to 0.5.

To run the parameter tuning. Do the following:

1. In tune.yml, set tuning to True.
2. In tune.yml, choose the dataset name (ds_name). Comment the other names. 
3. In tune.yml, choose the algo name. Comment the rest. (*note: currently, just chose torch-gd-accel*)
4. If you intend to tune parameters for object: obj_000001 in the dataset, run the following command
    
    ```python
    python tune_parameters.py tune.yml obj_000001
    ```

**Note: Running Parameter Tuning will overwrite the existing hyperparameter files in ./hyper_parameters/**


## Running Corrector Analysis

After tuning parameters, for an object, we can now run the corrector analysis. 

1. In parameters.yml, set visualize to False
2. In parameters.yml, choose the dataset name (ds_name). Comment the other names. 
3. In parametets.yml, choose the algo name. Comment the rest. (*note: currently, just chose torch-gd-accel*)
4. If you intend to run corrector analysis for object: obj_000001 in the dataset, run the following command
    
    ```python
    python analyze_corrector.py parameters.yml obj_000001
    ```

## Visualizing the Corrector Performance

After tuning parameters, for an object, we can now visualize the performance of the 
corrector. 

1. In parameters.yml, set visualize to True
2. In parameters.yml, choose the dataset name (ds_name). Comment the other names. 
3. In parametets.yml, choose the algo name. Comment the rest. (*note: currently, just chose torch-gd-accel*)
4. If you intend to visualize the corrector for object: obj_000001 in the dataset, run the following command
    
    ```python
    python analyze_corrector.py parameters.yml obj_000001
    ```

## RSS Figures
Run plot/plot_rss_figs.py.


## Zip figures for backup
Run
``` 
 zip -r "corrector-analysis-figures-$(date +"%Y%m%d%H%M").zip" ./figures
```