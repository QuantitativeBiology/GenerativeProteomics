# Hugging Face 

Hugging Face is a widely known and used platform in several fields. It offers the user access to a large range of pre-trained models and datasets.

That is why we intend to offer the possibility to use one of our pre-trained models available on 
Hugging Face to perform the imputation.
 
In here, you can test the usage of one of our pre-trained models.

The **hugging.py** file contains the code necessary to perform the download and to use the model.
You can also find the **hela_missing_dann.csv** dataset, which is the one that is going to be used for this  test.

## Instalation 

In order to start, you can use the following commands :

```bash 
pip install transformers
```

```bash 
pip install huggingface_hub
```
Once these packages are installed, it should be ready to run the model.

``` bash

    python hugging.py --model <model_name> 
```

where `<model_name>` is the name of the model you want to use.

The available models are the following:

1. GAIN_DANN_model

## Expected Output 

After downloading the pre-trained model from Hugging Face and using it, you can expect it to produce 2 diferent types of output.

1. **x_reconstructed.csv**: corresponds to the reconstructed version of the input data after imputation.

2. **x_domain.csv**: predicted domain for each sample in the input data, output by the domain classifier.