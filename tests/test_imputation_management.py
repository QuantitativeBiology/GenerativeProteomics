import unittest
import numpy as np
import torch
import random
import os
import sys 
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GenerativeProteomics")))

from GenerativeProteomics.imputation_management import Imputation_Management 
import GenerativeProteomics.utils 

class TestImputationManagement(unittest.TestCase):
    
    
    def test_correct_model(self):
        """test running a correct model"""

        self.seed = 42  
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        df_missing = pd.read_csv("hela_missing_dann.csv")
        imputation_management = Imputation_Management("GAIN_DANN_model", df_missing, "hela_missing_dann.csv")
        imputation_management.run_model("GAIN_DANN_model")
        self.assertTrue(os.path.isdir("GAIN_DANN_model"), "The directory does not exist")

    def test_incorrect_model(self):
        """test the class with an incorrect model"""
        df_missing = pd.read_csv("hela_missing_dann.csv")
        imputation_management = Imputation_Management("non_existing", df_missing, "hela_missing_dann.csv")
        with self.assertRaises(SystemExit):
            imputation_management.run_model("non_existing")
    
    def test_add_existing_model(self):
        """test adding an existing model"""
        df_missing = pd.read_csv("hela_missing_dann.csv")
        imputation_management = Imputation_Management("model_1", df_missing, "hela_missing_dann.csv")
        imputation_management.add_method("model_1", "model_1_function")
        self.assertEqual(imputation_management.dict_imputation_methods["model_1"], "model_1_function")

    def test_add_exhisting_model(self):
        """test to try adding model already known"""
        imputation_management = Imputation_Management("GAIN_DANN_model", None, "hela_missing_dann.csv")
        with self.assertRaises(SystemExit):
            imputation_management.add_method("GAIN_DANN_model", "hugging_face_gain_dann")
            

    def test_medium_imputation(self):
        """test the medium imputation method"""
        df_missing = pd.read_csv("breastMissing_20.csv")
        imputation_management = Imputation_Management("medium_imputation", df_missing, "breastMissing_20.csv")
        result = imputation_management.run_model("medium_imputation")
        file = pd.read_csv("output_medium.csv")
        np.testing.assert_allclose(result, file, rtol=1e-8, atol=1e-12, err_msg="There are still missing values after medium imputation")

if __name__ == '__main__':
    unittest.main()