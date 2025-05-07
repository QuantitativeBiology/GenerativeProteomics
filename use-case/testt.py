from GenerativeProteomics import utils
from GenerativeProteomics import Network
from GenerativeProteomics import Params
from GenerativeProteomics import Metrics
from GenerativeProteomics import Data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig
import torch
import argparse
import os


def test_network():
    """ Test to showcase how to import and use the classes and functions of the GenerativeProteomics package. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, help = "indicates the model to use")
    return parser.parse_args()

if __name__ == "__main__":
    # Load the dataset
    args = test_network()
    if args.model is None :
        dataset_path = "PXD004452-8c3d7d43-b1e7-4a36-a430-23e41bcbe07c.absolute.tsv"  # Input dataset with missing values
        ref_path = None  # Reference complete dataset
        
        # Load dataset and reference
        dataset_df = utils.build_protein_matrix(dataset_path)
        dataset = dataset_df.values  # Convert to numpy array

        
        # Extract headers (missing_header)
        missing_header = dataset_df.columns.tolist()

        # Define parameters for testing
        params = Params(
            input=dataset_path,
            output="imputed.csv",
            ref=ref_path,
            output_folder=".",
            num_iterations=2001,
            batch_size=128,
            alpha=10,
            miss_rate=0.1,
            hint_rate=0.9,
            lr_D=0.001,
            lr_G=0.001,
            override=1,
            output_all=1,
        )
        
        input_dim = dataset.shape[1]  # Number of features
        h_dim = input_dim  # Hidden layer size
        net_G = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, input_dim),
            torch.nn.Sigmoid()
        )
        net_D = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, input_dim),
            torch.nn.Sigmoid()
        )
        
        # Initialize metrics
        metrics = Metrics(params)

        # Initialize the Network
        network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

        # Initialize Data
        data = Data(
            dataset=dataset,
            miss_rate=0.2,
            hint_rate=0.9,
            ref = None  # Provide reference if available
        )
        
        # Perform training (imputation)
        print("Running imputation...")
        try:
            network.evaluate(data=data, missing_header=missing_header)  
            network.train(data=data, missing_header=missing_header) 
            print("Imputation completed successfully!")
        except Exception as e:
            print(f"Error during imputation: {e}")

    else:
        model_name =  args.model
        print(model_name)
        try:
            # Load the tokenizer and model
            #the tokenizer preprocesses the input text so it can be understood by the model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except (OSError, PretrainedConfig) as e:
            print(f"Could not load model: {e}")
            exit(1)

        # Save the model and tokenizer
        save_dir = "./saved_model"
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print(f"Model and tokenizer saved to {save_dir}")


        #-------------------------------- ADD CODE FOR PRE-TRAINED MODEL HERE ------------------------------






