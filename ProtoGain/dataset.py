import torch
import numpy as np
import pandas as pd

from missforest import MissForest

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Data:
    def __init__(self, dataset, miss_rate, hint_rate, ref=None, axis="columns"):

        self.ref_mean_imputed = None
        self.ref_mean_scaled = None

        # self.scaler = StandardScaler()
        # dataset_scaled = self.scaler.fit_transform(dataset)

        mask = np.where(np.isnan(dataset), 0.0, 1.0)
        # dataset = np.where(mask, dataset, 0.0)
        # dataset_scaled = np.where(mask, dataset_scaled, 0.0)

        ###### Initialize nans with mean of the protein
        dataset = pd.DataFrame(dataset)

        if axis == "columns":
            dataset = dataset.fillna(dataset.mean(axis=0))
            dataset = dataset.values
            missingness = 1 - mask.mean(axis=0)
        elif axis == "rows":
            dataset = dataset.T.fillna(dataset.mean(axis=1))
            dataset = dataset.T
            dataset = dataset.values
            missingness = 1 - mask.mean(axis=1)
        #########################################

        # MinMaxScaler

        self.scaler = MinMaxScaler()
        # self.scaler_T = MinMaxScaler()

        dataset_scaled = self.scaler.fit_transform(dataset)
        # dataset_scaled_T = self.scaler_T.fit_transform(dataset.T)

        hint = generate_hint(mask, hint_rate)

        self.dataset = torch.from_numpy(dataset)
        self.mask = torch.from_numpy(mask)
        self.hint = torch.from_numpy(hint)
        self.dataset_scaled = torch.from_numpy(dataset_scaled)
        self.missingness = torch.from_numpy(missingness)

        self.dataset_T = torch.from_numpy(dataset.T)
        self.mask_T = torch.from_numpy(mask.T)
        self.hint_T = torch.from_numpy(hint.T)
        self.dataset_scaled_T = torch.from_numpy(dataset_scaled.T)
        self.missingness_T = torch.from_numpy(missingness.T)

        if ref is not None:
            ref_mask = np.where(np.isnan(ref), 0.0, 1.0)
            ref_dataset = np.where(ref_mask, ref, 0.0)
            ref_hint = generate_hint(ref_mask, hint_rate)
            ref_dataset_scaled = self.scaler.transform(ref_dataset)
            ref_missingness = 1 - ref_mask.mean(axis=0)

            self.ref_dataset = torch.from_numpy(ref_dataset)
            self.ref_mask = torch.from_numpy(ref_mask)
            self.ref_hint = torch.from_numpy(ref_hint)
            self.ref_dataset_scaled = torch.from_numpy(ref_dataset_scaled)
            self.ref_missingness = torch.from_numpy(ref_missingness)
        else:
            self._create_ref(miss_rate, hint_rate, axis)

        print("\nNumber of samples:", self.dataset.shape[0])
        print("Number of features:", self.dataset.shape[1])
        print("Missing Rate (%):", (1.0 - self.mask.mean().item()) * 100.0)

    def _create_ref(cls, miss_rate, hint_rate, axis):

        cls.ref_mask = cls.mask.detach().clone()
        cls.ref_dataset = cls.dataset.detach().clone()
        zero_idxs = torch.nonzero(cls.mask == 1)
        chance = torch.rand(len(zero_idxs))
        miss = chance > miss_rate

        selected_idx = zero_idxs[~miss]
        for idx in selected_idx:
            cls.ref_mask[tuple(idx)] = 0
            cls.ref_dataset[tuple(idx)] = 0

        df_dummy = np.where(cls.ref_mask, cls.ref_dataset, np.nan)
        df_dummy = pd.DataFrame(df_dummy)

        if axis == "columns":
            df_dummy = df_dummy.fillna(df_dummy.mean(axis=0))
            cls.ref_missingness = 1 - cls.ref_mask.mean(axis=0)
        elif axis == "rows":
            df_dummy = df_dummy.T.fillna(df_dummy.mean(axis=1))
            df_dummy = df_dummy.T
            cls.ref_missingness = 1 - cls.ref_mask.mean(axis=1)

        cls.ref_mean_imputed = torch.tensor(df_dummy.values)

        cls.ref_hint = generate_hint(cls.ref_mask, hint_rate)
        cls.ref_dataset_scaled = torch.from_numpy(cls.scaler.transform(cls.ref_dataset))
        cls.ref_mean_scaled = torch.from_numpy(
            cls.scaler.transform(cls.ref_mean_imputed)
        )

        cls.ref_mean_imputed_T = cls.ref_mean_imputed.T
        cls.ref_mean_scaled_T = cls.ref_mean_scaled.T
        cls.ref_dataset_T = cls.ref_dataset.T
        cls.ref_mask_T = cls.ref_mask.T
        cls.ref_hint_T = cls.ref_hint.T
        cls.ref_dataset_scaled_T = cls.ref_dataset_scaled.T

        cls.ref_MF_imputed_T = cls.MF_impute()

        cls.ref_MF_scaled_T = torch.from_numpy(
            cls.scaler.transform(cls.ref_MF_imputed_T.T)
        ).T

        cls.ref_MF_imputed = cls.ref_MF_imputed_T.T
        cls.ref_MF_scaled = cls.ref_MF_scaled_T.T

    def MF_impute(cls):
        mf = MissForest()

        X = cls.ref_dataset_T.detach().numpy()
        X = np.where(cls.ref_mask_T == 0, np.nan, X)

        X = pd.DataFrame(X)

        print(X, cls.ref_mask_T)

        print("\nStarted MF\n")
        imputed = mf.fit_transform(X)

        print(imputed)
        print(torch.from_numpy(imputed.values))
        return torch.from_numpy(imputed.values)


def generate_hint(mask, hint_rate):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask

    return hint


def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    mask = 1.0 * B

    return mask


def generate_hint_missingness(mask, hint_rate, missingness):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask
    hint = np.multiply(hint, missingness)

    return hint


def generate_hint_paper(mask, hint_rate, missingness):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask + 0.5 * (1 - hint_mask)

    return hint


def generate_hint_paper_missingness(mask, hint_rate, missingness):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask + np.multiply((1 - missingness), (1 - hint_mask).T).T

    return hint
