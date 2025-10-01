from abc import ABC, abstractmethod

class ImputationModel(ABC):
    '''Abstract base class for imputation models.'''

    @abstractmethod
    def run(self, df):
        '''Run the imputation model on the provided DataFrame.'''
        pass
    
