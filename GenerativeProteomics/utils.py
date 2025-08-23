from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import pandas as pd
import os
import anndata as ad
import polars as pl
import json

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


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def build_protein_matrix(tsv_file):

    data = pd.read_csv(
        tsv_file,
        sep="\t",
        lineterminator="\n",
        skiprows=(10),
        header=(0),
        usecols=(0, 1, 4),
    )

    # matrix = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    matrix = data.pivot(index="protein", columns="sample_accession", values="ribaq")

    # print("Number of samples", matrix.shape[0])
    # print("Number of features", matrix.shape[1])
    # print("Missing Rate (%)", matrix.isna().sum().sum() / matrix.size * 100)

    return matrix


def build_protein_matrix_from_anndata(anndata_file: str)  -> pd.DataFrame:
    """
        An anndata file has the .h5ad extension.
    """

    adata = ad.read_h5ad(anndata_file)

    data = pd.DataFrame(
        adata.X,
        index=adata.obs["SampleID"],
        columns=adata.var["ProteinName"]
    )

    data = data.T

    return data

def handle_parquet(parquet_file):

    df = pl.read_parquet(parquet_file)
    #df_pandas = df.to_pandas()
    df_numeric = df.select(pl.col(pl.Float64, pl.Int64, pl.Int32, pl.Float32))

    #numeric_cols = df_pandas.select_dtypes(include=[np.number])
    #dataset = numeric_cols.to_numpy(dtype=np.float32)

    return df_numeric

def hugging_face(model, dataset_path):

    """ Function to import and use the classes and functions of the GenerativeProteomics package. """

    save_dir = "./GAIN_DANN_model"
    os.makedirs(save_dir, exist_ok=True)

    # Download files manually
    config_path = hf_hub_download(repo_id = f"QuantitativeBiology/{model}", filename="config.json", cache_dir = save_dir)
    weights_path = hf_hub_download(repo_id =f"QuantitativeBiology/{model}", filename="pytorch_model.bin", cache_dir = save_dir)
    model_path = hf_hub_download(repo_id = f"QuantitativeBiology/{model}", filename="modeling_gain_dann.py", cache_dir = save_dir)

    print("config_path:", config_path)
    print("weights_path:", weights_path)
    print("model_path", model_path)

    directory = os.path.dirname(model_path)
    print("directory:", directory)

    import sys
    sys.path.append(directory)  # Add the directory containing 'modeling_gain_dann.py' to the Python path

    if model == "GAIN_DANN_model":

    # Now you can import the functions or classes inside the file
        from modeling_gain_dann import GainDANNConfig, GainDANN


        print("import successfully done")

        # Load config
        with open(config_path) as f:
            cfg = json.load(f)

        hela = pd.read_csv(dataset_path, index_col=0)
        #hela_numeric = hela.drop(columns=['Project'])

        input_dim = hela.shape[1]
        
        print(input_dim)
        cfg['input_dim'] = input_dim

        config = GainDANNConfig(**cfg)
        model = GainDANN(config)

        # Load raw state_dict
        state_dict = torch.load(weights_path, map_location="cpu")

        # Add "model." prefix to every key(because state_dict has the weights of GAIN_DANN but I added the Gain_DANN for the huggingFace accordance)
        renamed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        # Load with corrected keys
        model.load_state_dict(renamed_state_dict)

        model.eval()

        print(hela.dtypes)

        label_encoder = LabelEncoder()
        hela['Project'] = label_encoder.fit_transform(hela['Project'])

        x = torch.tensor(hela.values, dtype=torch.float32)

        with torch.no_grad():
            x_reconstructed, x_domain = model(x)

        print("x_reconstructed:", x_reconstructed)
        print("x_domain:", x_domain)

        pd.DataFrame(x_reconstructed.numpy()).to_csv("x_reconstructed.csv", index=False)
        pd.DataFrame(x_domain.numpy()).to_csv("x_domain.csv", index=False)
