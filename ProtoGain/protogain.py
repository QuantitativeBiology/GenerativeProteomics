from hypers import Params
from model import Network
from dataset import Data
from output import Metrics
import utils

import torch
from torch import nn
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm
import pandas as pd


import optuna

import time
import cProfile
import pstats
import argparse
import os
import psutil


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path to missing data")
    parser.add_argument("-o", default="imputed", help="name of output file")
    parser.add_argument("--ref", help="path to a reference (complete) dataset")
    parser.add_argument("--oeval", help="name of output file for evaluation")
    parser.add_argument(
        "--ofolder", default=os.getcwd() + "/results/", help="path to output folder"
    )
    parser.add_argument("--it", type=int, default=2001, help="number of iterations")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--alpha", type=float, default=10, help="alpha")
    parser.add_argument("--miss", type=float, default=0.2, help="missing rate")
    parser.add_argument("--hint", type=float, default=0.8, help="hint rate")
    parser.add_argument(
        "--trainratio", help="percentage of data to be used as a train set"
    )
    parser.add_argument(
        "--lrd", type=float, default=0.001, help="learning rate for the discriminator"
    )
    parser.add_argument(
        "--lrg", type=float, default=0.001, help="learning rate for the generator"
    )
    parser.add_argument("--parameters", help="load a parameters.json file")
    parser.add_argument(
        "--override", type=int, default=0, help="override previous files"
    )
    parser.add_argument("--outall", type=int, default=0, help="output all files")

    ##### Specific for multiple_runs Yasset #####
    parser.add_argument("--project", help="project name")

    parser.add_argument("--cores", type=int, default=1, help="number of cores")

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    # num_processes = 4

    with cProfile.Profile() as profile:

        folder = os.getcwd()

        args = init_arg()

        missing_file = args.i
        output_file = args.o
        ref_file = args.ref
        output_eval = args.oeval
        output_folder = args.ofolder
        num_iterations = args.it
        batch_size = args.batchsize
        alpha = args.alpha
        miss_rate = args.miss
        hint_rate = args.hint
        train_ratio = args.trainratio
        lr_D = args.lrd
        lr_G = args.lrg
        parameters_file = args.parameters
        override = args.override
        output_all = args.outall

        project = args.project
        cores = args.cores

        torch.set_num_threads(cores)
        torch.set_num_interop_threads(cores)

        print(torch.get_num_threads())
        print(torch.get_num_interop_threads())

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Using CPU")

        if parameters_file is not None:
            params = Params.read_hyperparameters(parameters_file)
            missing_file = params.input
            output_file = params.output
            ref_file = params.ref
            output_eval = params.output_eval
            output_folder = params.output_folder
            num_iterations = params.num_iterations
            batch_size = params.batch_size
            alpha = params.alpha
            miss_rate = params.miss_rate
            hint_rate = params.hint_rate
            train_ration = params.train_ratio
            lr_D = params.lr_D
            lr_G = params.lr_G
            override = params.override
            output_all = params.output_all

        else:
            params = Params(
                missing_file,
                output_file,
                ref_file,
                output_eval,
                output_folder,
                num_iterations,
                batch_size,
                alpha,
                miss_rate,
                hint_rate,
                train_ratio,
                lr_D,
                lr_G,
                override,
                output_all,
            )

        if not os.path.exists(params.output_folder):
            os.makedirs(params.output_folder)

        if missing_file is None:
            print("Input file not provided")
            exit(1)
        if missing_file.endswith(".csv"):
            df_missing = pd.read_csv(missing_file)
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
        elif missing_file.endswith(".tsv"):
            df_missing = utils.build_protein_matrix(missing_file, "columns")
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
            missing_index = df_missing.index.tolist()

            if not os.path.exists(f"{output_folder}{project}.csv"):
                utils.create_csv(missing, f"{output_folder}{project}", missing_header)
        else:
            print("Invalid file format")
            exit(2)

        train_size = missing.shape[0]
        dim = missing.shape[1]

        h_dim1 = dim
        h_dim2 = dim

        net_G = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        net_D = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        # net_G.share_memory()
        # net_D.share_memory()

        metrics = Metrics(params)
        model = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

        if ref_file is not None:
            df_ref = pd.read_csv(ref_file)
            ref = df_ref.values
            ref_header = df_ref.columns.tolist()

            if dim != ref.shape[1]:
                print(
                    "\n\nThe reference and data files provided don't have the same number of features\n"
                )
                exit(3.1)
            elif train_size != ref.shape[0]:
                print(
                    "\n\nThe reference and data files provided don't have the same number of samples\n"
                )
                exit(3.2)

            data = Data(missing, miss_rate, hint_rate, ref)
            model.train_ref(data, missing_header)

        else:

            data = Data(missing, miss_rate, hint_rate, axis="columns")

            # model.evaluate(data, missing_header, transpose=0)
            # model.train(data, missing_header, transpose=0)

            ################ Starting Transpose run ################

            print("\n\nStarting Transpose run\n")

            dim = missing.shape[0]
            h_dim1 = dim
            h_dim2 = dim

            dummy = missing_header
            missing_header = missing_index
            missing_index = dummy

            net_G = nn.Sequential(
                nn.Linear(dim * 2, h_dim1),
                nn.ReLU(),
                nn.Linear(h_dim1, h_dim2),
                nn.ReLU(),
                nn.Linear(h_dim2, dim),
                nn.Sigmoid(),
            )

            net_D = nn.Sequential(
                nn.Linear(dim * 2, h_dim1),
                nn.ReLU(),
                nn.Linear(h_dim1, h_dim2),
                nn.ReLU(),
                nn.Linear(h_dim2, dim),
                nn.Sigmoid(),
            )

            metrics = Metrics(params)
            model = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

            model.evaluate(data, missing_header, transpose=1)

            #########################################################################

        if project is not None:
            file_path = output_folder + "dataset_descriptions.csv"
            if os.path.exists(file_path):

                df = pd.read_csv(file_path)
                if project not in df["dataset"].values:

                    with open(file_path, "a") as myfile:
                        myfile.write(
                            f"{project},{data.dataset.shape[0]},{data.dataset.shape[1]},{(1.0 - data.mask.mean().item()) * 100.0}"
                            + "\n"
                        )

            else:
                with open(file_path, "w") as myfile:
                    myfile.write("dataset,nsamples,nfeatures,missingrate\n")
                    myfile.write(
                        f"{project},{data.dataset.shape[0]},{data.dataset.shape[1]},{(1.0 - data.mask.mean().item()) * 100.0}"
                        + "\n"
                    )

        run_time = []
        run_time.append(time.time() - start_time)
        file_path = output_folder + "run_time.csv"

        if override == 1:
            df_run_time = pd.DataFrame(run_time)
            df_run_time.to_csv(file_path, index=False)

        else:
            if os.path.exists(file_path):
                with open(file_path, "a") as myfile:
                    myfile.write(str(run_time[0]) + "\n")

            else:
                df_run_time = pd.DataFrame(run_time)
                df_run_time.to_csv(file_path, index=False)

    print("\n--- %s seconds ---\n\n" % (run_time[0]))
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    results.dump_stats("results.prof")
