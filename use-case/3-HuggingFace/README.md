# HuggingFace 

HuggingFace is a widely known and used platform in several fields. It offers the user access to a large range of pre-trained models and datasets.

That is why we intend to offer the possibility to use one of our pre-trained models available on 
HuggingFace to perform the imputation.
 
In here, you can test the usage of one of our pre-trained models.


## Installation 

In order to start, you can use the following commands to install the necessary packages:

```bash 
pip install transformers
```

```bash 
pip install huggingface_hub
```

Once these packages are installed, it should be ready to run the model.

## Input 

To run the pre-trained model, you need to specify the model name you wish to use with the `--model` flag when running the script.
You should also provide a dataset that has missing values, which will be imputed by one of the models we have to offer.

### Available models

1. `GAIN_DANN_model`


## Example

In this use-case, you can find a file that shows how to download the model from HuggingFace and how to use it to perform imputation.

The **hugging.py** file contains all the code necessary to perform the download and to use the model.

You can also find the **hela_missing_dann.csv** dataset, which is the one that is going to be used for this test.

In this example, we will be using the `GAIN_DANN_model`.

To run the file, all you have to do is run the following command:

``` bash

    python hugging.py --model GAIN_DANN_model 
```

## Expected Output 

After downloading the pre-trained model from Hugging Face and using it, you can expect it to produce two diferent types of output.

1. **x_reconstructed.csv**: 
- Corresponds to the reconstructed version of the input data after imputation.
- Each value corresponds to the model's best estimates.

2. **x_domain.csv**: 
- Predicted domain for each sample in the input data, outputed by the domain classifier.