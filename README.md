# Generative Proteomics

[![PyPi Version](https://img.shields.io/pypi/v/GenerativeProteomics?label=PyPi&color=blue&style=flat&logo=pypi)](https://pypi.org/project/GenerativeProteomics/)
[![Colab](https://img.shields.io/badge/Google_Colab-0061F2?style=flat&logo=googlecolab&color=blue&label=Colab&colorB=grey)](https://colab.research.google.com/drive/1a-DfiAtPqmU2AnfMS-plDuTkZOfQ_zrB?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue)](https://generativeproteomics.readthedocs.io/en/latest/)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-grey?style=flat&logo=huggingface&color=grey)](https://huggingface.co/QuantitativeBiology)

In this repository you may find a PyTorch implementation of Generative Adversarial Imputation Networks (GAIN) [[1]](#1) for imputing missing iBAQ values in proteomics datasets.

## Table of Contents

- [Installation](#installation)
- [How to Use](#usage)
- [Demo](#demo)
- [References](#reference)

## Installation

### Pip install

We have submitted a package to the Python Package Index (PyPI) for easy installation. You can install the package using the following command:

```bash
pip install GenerativeProteomics
```

This way, you can install the package and its dependencies in one go.

After that, you can import all the functions and classes from the package of the model and use them in your code.

### GitHub

If you prefer to use the code of the GenerativeProteomics model directly, you can access it in our GitHub repository and follow the next sequence of commands.

1. Clone this repository:  `git clone https://github.com/QuantitativeBiology/GenerativeProteomics/`
2. Create a Python environment: `conda create -n proto python=3.10` if you have conda installed
3. Activate the previously created environment: `conda activate proto`
4. Install the necessary packages: `pip install -r libraries.txt`


## How to Use

If you just want to impute a general dataset, the most straightforward and simplest way to run GenerativeProteomics is to run: `python generativeproteomics.py -i /path/to/file_to_impute.csv`
Running in this manner will result in two separate training phases.

1) Evaluation run: In this run a percentage of the values (10% by default) are concealed during the training phase and then the dataset is imputed. The RMSE is calculated with those hidden values as targets and at the end of the training phase a `test_imputed.csv` file will be created containing the original hidden values and the resulting imputation, this way you can have an estimation of the imputation accuracy.

2) Imputation run: Then a proper training phase takes place using the entire dataset. An `imputed.csv` file will be created containing the imputed dataset.


However, there are a few arguments which you may want to change. You can do this using a parameters.json file (you may find an example in `GenerativeProteomics/breast/parameters.json`) or you can choose them directly in the command line.

Run with a parameters.json file: `python generativeproteomics.py --parameters /path/to/parameters.json`<br>
Run with command line arguments: `python generativeproteomics.py -i /path/to/file_to_impute.csv -o imputed_name --ofolder ./results/ --it 2001`

#### Arguments:

`-i`: Path to file to impute<br> 
`-o`: Name of imputed file<br> 
`--ofolder`: Path to the output folder<br> 
`--it`: Number of iterations to train the model<br> 
`--miss`: The percentage of values to be concealed during the evaluation run (from `0` to `1`)<br>
`--outall`: Set this argument to `1` if you want to output every metric<br> 
`--override`: Set this argument to `1` if you want to delete the previously created files when writing the new output<br> 



If you want to test the efficacy of the code you may give a reference file containing a complete version of the dataset (without missing values): `python generativeproteomics.py -i /path/to/file_to_impute.csv --ref /path/to/complete_dataset.csv`

Running this way will calculate the RMSE of the imputation in relation to the complete dataset.


## Demo

In this repository you may find a folder named `breast`, inside it you have a breast cancer diagnostic dataset [[2]](#2) which you may use to try out the code.

`breast.csv`: complete dataset<br>
`breastMissing_20.csv`: the same dataset but with 20% of its values taken out


To simply impute `breastMissing_20.csv` run: `python generativeproteomics.py -i ./breast/breastMissing_20.csv` <br>
If you want to compare the imputation with the original dataset run: `python generativeproteomics.py -i ./breast/breastMissing_20.csv --ref ./breast/breast.csv` or `python generativeproteomics.py --parameters ./breast/parameters.json`


If you want to go deep in the analysis of every metric you either set `--outall` to `1` or you run the code in an IPython console, this way you can access every variable you want in the `metrics` object, e.g. `metrics.loss_D`.


## References
<a id="1">[1]</a> 
J. Yoon, J. Jordon & M. van der Schaar (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets <br>
<a id="2">[2]</a> 
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
