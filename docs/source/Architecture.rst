.. _architecture:

Architecture
===============

GenerativeProteomics follows a modular architecture that promotes flexibility and scalability.
It is composed of six main classes responsible for different tasks in the processing and imputation of large proteomics datasets.
Among those tasks, we can highlight the data processing, the imputation of missing values, the generation of synthetic data and the metrics calculation.

Bellow, you can find a class diagram that showcases how these modules are connected and how they interact with each other.

.. image:: _static/class_diagram_no_label.drawio.png
    :alt: class_diagram_abstract
    :width: 500px
    :align: center

Overview
--------

This diagram illustrates the overall architecture of GenerativeProteomics, showing how the different components interact during the imputation process.

- **GenerativeProteomics**: The main entry point that initializes all classes.
- **Data**: Handles dataset loading and preparation, as well as the creation of the hint matrix, the mask matrix and the synthetic reference dataset.
- **Network**: Trains the model using the attributes from the `Data` class.
- **Params**: Stores hyperparameters and passes them to the network.
- **Metrics**: Computes key metrics such as loss values for both the Discriminator and Generator.
- **Utils**: Provides auxiliary functions like indexing, output generation, and CSV creation.

Execution Flow   
--------------

1. The `GenerativeProteomics` module orchestrates the imputation process.
2. The `Data` module loads the dataset, which is used by the `Network`.
3. The `Network` requires hyperparameters from the `Params` class.
4. The `Metrics` class contains evaluation metrics from the training process.
5. The `Utils` class provides helper functions for tasks like file management.
6. The model outputs files such as `impute.csv`, `test_imputed.csv`, and performance metrics like `loss_G` and `loss_D`.

This structure ensures **modularity, maintainability, and scalability**, making it easier to extend GenerativeProteomics.
