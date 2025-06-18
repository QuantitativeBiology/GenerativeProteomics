# Generative Proteomics Tests

In this directory, you can find the tests for the GenerativeProteomics model.

## Tests 

- `test_imputation_with_reference`: test to infer the imputation process with a reference dataset
- `test_imputate_no_reference`: test to infer the imputation process with a reference dataset
- `test_hyper`: tests if the model is correctly reading and updating the parameters used for imputation
- `test_hint_generation`: test to infer the process of generating the hint matrix
- `test_generate_reference`: test to infer the production of a synthetic reference dataset

## Test Goals

These tests were developed to ensure the proper functioning of the model.

They are particularly useful when you change a piece of code and you want to determine if the model is still showcasing the proper behaviour.

## How they work 

Tests use PRIDE datasets with reproducible random seeds in order to ensure consistent results:

1. Create a unittest class where the tests will take place
2. Execute the task that is being tested
3. Compare the output with previously saved results

## Commands

To run a specific test, use: 
```bash 
python -m unittest <name_of_test>
```

If you want to run all tests at once, use:
```bash 
python -m unittest discover
```

Additionally, if you want to check the coverage of the tests too, you can use the `coverage` package.

First, you install it with:
```bash 
pip install converage
```

Then, to use it on a specific test, run the following command:

```bash 
coverage run -m unittest <name_of_test>
```
If you want to run them all, run the following:
```bash 
coverage run -m unittest discover
```

After running the tests, to check the coverage run the dollowing command:
```bash 
coverage report
```

For a more detailed insight, use this command, which will create and html file that you can open in your browser:
```bash
coverage html
```
