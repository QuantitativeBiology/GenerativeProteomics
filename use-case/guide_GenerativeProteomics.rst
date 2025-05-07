GenerativeProteomics
====================

In this use case, we will show in a simple and clear way how to install and use `GenerativeProteomics` 
to perform imputation of missing values of proteomics' datasets.

Installation
-------------

`GenerativeProteomics` is a Python package for imputation of missing values in the field of proteomics. 
It is currently based on the `Generative Adversarial Imputation Network (GAIN)` architecture.
To use the package, you need to have `Python 3.10` or `Python 3.11` on your system.
The package is available on `PyPI` and can be installed using a `pip` command.

.. code-block:: bash

    pip install GenerativeProteomics     


By running the pip command, you are also installing all the dependencies required by the package, 
which are the following :

- **torch**
- **torchinfo**
- **numpy**
- **tqdm**
- **pandas**
- **scikit-learn**
- **optuna**
- **argparse**
- **psutil**

Contents
--------

The package allows you to import and use the following classes, with each one of them having a specific 
role in the imputation process :

1. Data
    - handles datasets with missing values. It preprocesses the dataset, generates necessary masks, and 
    scales the data for model training
    - provides functions like:
            - generate_hint()
            - generate_mask()
            - _create_ref()

2. Metrics
    - tracks performance metrics during the training and evaluation of the model
    - provides functions like:
            - _create_output()

3. Network
    - defines the architecture of the Generator and Discriminator networks
    - provides functions like:
            - generate_sample()
            - impute()
            - evaluate_impute()
            - update_D()
            - update_G()
            - train_ref()
            - evaluate()
            - train()

4. Param
    - contains the hyperparameters of the model
    - provides functions like:
            - read_json()
            - read_hyperparameters()
            - _update_hypers()

5. utils
    - contains utility functions for the model
    - provides functions like:
            - create_csv()
            - create_dist()
            - create_missing()
            - create_output()
            - output()
            - sample_idx()
            - build_protein_matrix()


Example
=======

In this use-case, you can find a file that showcases how to import and use the functions and classes of GenerativeProteomics.
This file is called `testt.py` and it performs the imputation of missing values on a dataset of proteins from PRIDE.
The dataset in question is called `PXD004452.tsv` and it is also accessible in this directory. 
This dataset has a missing rate of 17.442532054984405%, 8657 samples and 4 features.

To run the file with the PRIDE dataset, you can use the following command :

.. code-block:: bash

    python testt.py 


Expected Output
---------------

The imputation model produces several forms of output.
Throughout the imputation process, the model updates on the terminal the progress of the process and 
the loss values of both the Discriminator and Generator.

It produces an `imputed.csv` file, which contains the imputed dataset.
Additionally, it also produces other csv files with information about the loss values of the Discriminator and Generator, 
as well as the metrics of the imputation process.

In the end, you should have access to the following files:

    - imputed.csv
    - loss_D.csv
    - loss_G.csv
    - lossMSE_test.csv
    - lossMSE_train.csv
    - cpu.csv
    - ram.csv 
    - ram_percentage.csv


Hugging Face
-------------

Additionally, we also offer the possibility to use one of our pre-trained models available on 
Hugging Face to perform the imputation.

In order to do so, you can use the following command :

.. code-block:: bash

    python testt.py --model_name_or_path <model_name> 


where `<model_name>` is the name of the model you want to use.

The available models are the following:
    - write down the names of our models available on Hugging Face

Using these models can have meaningful advantages, as they promote  
lower computation costs and do not take as much time to run.

Further Information
-------------------

For more information about our work, you can visit the following documentation:

https://generativeproteomics.readthedocs.io/en/latest/index.html