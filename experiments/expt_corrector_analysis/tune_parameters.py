import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

sys.path.append("./")
from corrector_analysis import run_full_experiment
from plot import make_plots, evaluate_for_hyperparameter_tuning

HOME_DIR = Path(__file__).parent

def tune_parameters(parameters,
                    epsilon_range,
                    clamp_thres_range):

    eval_metric = torch.ones(len(clamp_thres_range), len(epsilon_range))

    for idx_cl, clamp_thres in enumerate(clamp_thres_range):

        print("Running model with clamp_thres: ", clamp_thres)

        #note: this output does not depend on epsilon, which is used only in the evaluation stage.
        filenames = run_full_experiment(ds_names=parameters['ds_names'],
                                        kp_noise_fra=parameters['kp_noise_fra'],
                                        algo=parameters['algo'],
                                        clamp_thres=clamp_thres,
                                        epsilon=epsilon_range[0],
                                        corrector_max_solve_iters=parameters['corrector_max_solve_iters'],
                                        visible_fraction_ub=parameters['visible_fraction_ub'],
                                        visible_fraction_lb=parameters['visible_fraction_lb'],
                                        kp_noise_var_range=parameters['kp_noise_var_range'],
                                        object_labels=parameters['object_labels'],
                                        use_random=False,
                                        only_visualize=False,
                                        do_certification=False)

        for idx_ep, epsilon in enumerate(epsilon_range):

            Rerr, Rerr_cert, fra_cert = evaluate_for_hyperparameter_tuning(filenames, epsilon=epsilon)

            #ToDo: now using Rerr and not Rerr_cert
            if torch.isnan(Rerr[0]):
                metric_ = torch.tensor([float("Inf")])[0]
            else:
                metric_ = Rerr[0]

            eval_metric[idx_cl, idx_ep] = metric_

    val_, idx_cl_ = torch.min(eval_metric, dim=0)
    val, idx_ep_opt = torch.min(val_, dim=0)
    idx_cl_opt = idx_cl_[idx_ep_opt]

    epsilon_opt = epsilon_range[idx_ep_opt]
    clamp_thres_opt = clamp_thres_range[idx_cl_opt]
    Rerr_cert_opt = val

    return epsilon_opt.to(device='cpu').numpy(), clamp_thres_opt.to(device='cpu').numpy(), Rerr_cert_opt.to(device='cpu').numpy()


if __name__ == "__main__":
    """
       usage:
       >> python tune_parameters.py tune.yml obj_000001
       """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        help="name of the parameter file",
        default="tune.yml",
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
    parameters['object_labels'] = [args.object_label]

    if parameters['algo'] == 'torch-gd-accel':
        epsilon_range = torch.arange(0.002, 0.02, 0.002)    # 10
        clamp_thres_range = torch.arange(0.005, 0.05, 0.005)   # 10
        # clamp_thres_range = torch.arange(0.0450, 0.10, 0.005)
        # epsilon_range = torch.arange(0.002, 0.0042, 0.002)  # 10
        # clamp_thres_range = torch.arange(0.01, 0.07, 0.05)  # 10

        epsilon, clamp_thres, Rerr_cert_opt = tune_parameters(parameters,
                                                              epsilon_range,
                                                              clamp_thres_range)

        # breakpoint()
        data_dict = {}
        data_dict['ds_name'] = parameters['ds_names'][0]
        data_dict['object_label'] = args.object_label
        data_dict['algo'] = parameters['algo']
        data_dict['epsilon'] = float(epsilon)
        data_dict['clamp_thres'] = float(clamp_thres)
        data_dict['Rerr_cert_opt'] = float(Rerr_cert_opt)

        location = HOME_DIR / 'hyper_parameters'
        if not location.exists():
            os.makedirs(location)
        filename = parameters['ds_names'][0] + '.' + args.object_label + '.' + parameters['algo'] + '.yaml'

        with open(location / filename, 'w') as file:
            documents = yaml.dump(data_dict, file)

    elif parameters['algo'] == 'torch-mutlithres-gd-accel':
        print("Not implemented.")

    elif parameters['algo'] == 'torch-gnc-gm':
        print("Not implemented.")

    elif parameters['algo'] == 'torch-gnc-tls':
        print("Not implemented.")

    else:
        print("Not implemented.")
