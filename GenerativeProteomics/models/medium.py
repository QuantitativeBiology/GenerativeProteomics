from .base_abstract import ImputationModel
import numpy as np

class MediumImputationModel(ImputationModel):
    '''Imputation model using medium value for imputation.'''

    def run(self, df):
        '''Function that calculates the medium value of each column and imputes missing values with 
        that value '''
        df_2 = df.copy()
        num_cols = df_2.select_dtypes(include=[np.number]).columns

        df_2[num_cols] = df_2[num_cols].astype(float).replace(0, np.nan)

        col_means = df_2[num_cols].mean().fillna(0.0)

        df_2[num_cols] = df_2[num_cols].fillna(col_means)
        print(df_2)
        return df_2