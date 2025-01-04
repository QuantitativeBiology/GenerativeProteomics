import torch
import numpy as np
import pandas as pd
import os


def create_csv(data, name: str, header):
    df = pd.DataFrame(data)
    df.to_csv(name + ".csv", index=False, header=header)


def create_dist(size: int, dim: int, name: str):

    X = torch.normal(0.0, 1, (size, dim))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([0, 0])
    data = torch.matmul(X, A) + b

    create_csv(data, name)


def create_missing(data, miss_rate: float, name: str, header):

    size = data.shape[0]
    dim = data.shape[1]

    mask = torch.zeros(data.shape)

    for i in range(dim):

        chance = torch.rand(size)
        miss = chance > miss_rate
        mask[:, i] = miss

        missing_data = np.where(mask < 1, np.nan, data)

    name = name + "_{}".format(int(miss_rate * 100))

    create_csv(missing_data, name, header)


def create_output(data, path: str, override: int):

    if override == 1:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    else:
        if os.path.exists(path):
            df = pd.read_csv(path)
            new_df = pd.DataFrame(data)
            df = pd.concat([df, new_df], axis=1)
            df.columns = range(len(df.columns))
            df.to_csv(path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)


def output(
    data_train_imputed,
    output_folder,
    output_file,
    missing_header,
    loss_D_values,
    loss_G_values,
    loss_MSE_train,
    loss_MSE_test,
    cpu,
    ram,
    ram_percentage,
    override,
):

    create_csv(
        data_train_imputed,
        output_folder + output_file,
        missing_header,
    )
    create_output(
        loss_D_values,
        output_folder + "lossD.csv",
        override,
    )
    create_output(
        loss_G_values,
        output_folder + "lossG.csv",
        override,
    )
    create_output(
        loss_MSE_train,
        output_folder + "lossMSE_train.csv",
        override,
    )

    create_output(
        loss_MSE_test,
        output_folder + "lossMSE_test.csv",
        override,
    )

    create_output(
        cpu,
        output_folder + "cpu.csv",
        override,
    )

    create_output(ram, output_folder + "ram.csv", override)

    create_output(
        ram_percentage,
        output_folder + "ram_percentage.csv",
        override,
    )


def output_eval(
    data_train_imputed,
    output_folder,
    output_file,
    missing_header,
    loss_D_values,
    loss_G_values,
    loss_MSE_train,
    loss_MSE_test,
    cpu,
    ram,
    ram_percentage,
    override,
):

    create_csv(
        data_train_imputed,
        output_folder + output_file,
        missing_header,
    )
    create_output(
        loss_D_values,
        output_folder + "lossD_eval.csv",
        override,
    )
    create_output(
        loss_G_values,
        output_folder + "lossG_eval.csv",
        override,
    )
    create_output(
        loss_MSE_train,
        output_folder + "lossMSE_train_eval.csv",
        override,
    )

    create_output(
        loss_MSE_test,
        output_folder + "lossMSE_test_eval.csv",
        override,
    )

    create_output(
        cpu,
        output_folder + "cpu_eval.csv",
        override,
    )

    create_output(ram, output_folder + "ram_eval.csv", override)

    create_output(
        ram_percentage,
        output_folder + "ram_percentage_eval.csv",
        override,
    )


def output_eval_transpose(
    data_train_imputed,
    output_folder,
    output_file,
    missing_header,
    loss_D_values,
    loss_G_values,
    loss_MSE_train,
    loss_MSE_test,
    cpu,
    ram,
    ram_percentage,
    override,
):

    create_csv(
        data_train_imputed,
        output_folder + output_file + "_transpose",
        missing_header,
    )
    create_output(
        loss_D_values,
        output_folder + "lossD_eval_transpose.csv",
        override,
    )
    create_output(
        loss_G_values,
        output_folder + "lossG_eval_transpose.csv",
        override,
    )
    create_output(
        loss_MSE_train,
        output_folder + "lossMSE_train_eval_transpose.csv",
        override,
    )

    create_output(
        loss_MSE_test,
        output_folder + "lossMSE_test_eval_transpose.csv",
        override,
    )

    create_output(
        cpu,
        output_folder + "cpu_eval_transpose.csv",
        override,
    )

    create_output(ram, output_folder + "ram_eval_transpose.csv", override)

    create_output(
        ram_percentage,
        output_folder + "ram_percentage_eval_transpose.csv",
        override,
    )


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def build_protein_matrix(tsv_file, axis="columns"):

    data = pd.read_csv(
        tsv_file,
        sep="\t",
        lineterminator="\n",
        comment="#",
        header=(0),
        usecols=(0, 1, 4),
    )

    if axis == "columns":
        matrix = data.pivot(index="sample_accession", columns="protein", values="ribaq")

    elif axis == "rows":
        matrix = data.pivot(index="protein", columns="sample_accession", values="ribaq")

    # print("Number of samples", matrix.shape[0])
    # print("Number of features", matrix.shape[1])
    # print("Missing Rate (%)", matrix.isna().sum().sum() / matrix.size * 100)

    return matrix


def check_for_tsv_files(path):
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            for file in os.listdir(os.path.join(path, folder)):
                if file.endswith("absolute.tsv"):
                    print(file)
