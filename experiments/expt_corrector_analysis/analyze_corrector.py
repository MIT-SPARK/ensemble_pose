
import argparse
import sys
import yaml
from pathlib import Path

sys.path.append("./")
from corrector_analysis import run_full_experiment
from plot.plot import make_plots

HOME_DIR = Path(__file__).parent


if __name__ == "__main__":
    """
    usage:
    >> python analyze_corrector.py parameters.yml obj_000001
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        help="name of the parameter file",
        default="parameters.yml",
        type=str
    )

    parser.add_argument(
        "object_label",
        help="object label in the dataset",
        type=str
    )

    args = parser.parse_args()

    # unpacking parameters
    stream = open(args.file, "r")
    parameters = yaml.load(stream=stream, Loader=yaml.Loader)

    # making sure the parameters are correct
    assert parameters['visualize'] in [True, False]
    parameters['object_labels'] = [args.object_label]

    ## getting hyper-parameters
    #location = HOME_DIR / 'hyper_parameters'
    #filename = parameters['ds_names'][0] + '.' + parameters['object_labels'][0] + '.' + 'torch-gd-accel' + '.yaml'
    ## filename = parameters['ds_names'][0] + '.' + parameters['object_labels'][0] + '.' + parameters['algo'] + '.yaml'

    #fp = open(location / filename, 'r')
    #dict_ = yaml.load(stream=fp, Loader=yaml.Loader)
    # parameters['epsilon'] = dict_['epsilon']      # we use a uniform 2% epsilon for analysis
    # parameters['clamp_thres'] = dict_['clamp_thres']
    # parameters['clamp_thres'] = 0.10

    # analysis and plots
    if parameters['visualize']:
        filenames = run_full_experiment(ds_names=parameters['ds_names'],
                                        kp_noise_fra=parameters['kp_noise_fra'],
                                        algo=parameters['algo'],
                                        clamp_thres=parameters['clamp_thres'],
                                        epsilon=parameters['epsilon'],
                                        corrector_max_solve_iters=parameters['corrector_max_solve_iters'],
                                        visible_fraction_ub=parameters['visible_fraction_ub'],
                                        visible_fraction_lb=parameters['visible_fraction_lb'],
                                        kp_noise_var_range=parameters['kp_noise_var_range'],
                                        object_labels=parameters['object_labels'],
                                        use_random=False,
                                        only_visualize=parameters['visualize'],
                                        do_certification=False)

    else:
        filenames = run_full_experiment(ds_names=parameters['ds_names'],
                                        kp_noise_fra=parameters['kp_noise_fra'],
                                        algo=parameters['algo'],
                                        clamp_thres=parameters['clamp_thres'],
                                        epsilon=parameters['epsilon'],
                                        corrector_max_solve_iters=parameters['corrector_max_solve_iters'],
                                        visible_fraction_ub=parameters['visible_fraction_ub'],
                                        visible_fraction_lb=parameters['visible_fraction_lb'],
                                        kp_noise_var_range=parameters['kp_noise_var_range'],
                                        object_labels=parameters['object_labels'],
                                        use_random=False,
                                        only_visualize=False,
                                        do_certification=False)
        make_plots(filenames, epsilon=parameters['epsilon'])





