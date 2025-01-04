from hypers import Params
from dataset import Data
from output import Metrics

from dataset import generate_hint
from dataset import generate_hint_paper_missingness

import torch
from torch import nn
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import utils
import psutil

from torchinfo import summary


class Network:
    def __init__(self, hypers: Params, net_G, net_D, metrics: Metrics):

        # for w in net_D.parameters():
        #    nn.init.normal_(w, 0, 0.02)
        # for w in net_G.parameters():
        #    nn.init.normal_(w, 0, 0.02)

        # for w in net_D.parameters():
        #    nn.init.xavier_normal_(w)
        # for w in net_G.parameters():
        #    nn.init.xavier_normal_(w)

        for name, param in net_D.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        for name, param in net_G.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        self.hypers = hypers
        self.net_G = net_G
        self.net_D = net_D
        self.metrics = metrics

        self.optimizer_D = torch.optim.Adam(net_D.parameters(), lr=hypers.lr_D)
        self.optimizer_G = torch.optim.Adam(net_G.parameters(), lr=hypers.lr_G)

        # print(summary(net_G))

    def generate_sample(cls, data, mask):
        dim = data.shape[1]
        size = data.shape[0]

        Z = torch.rand((size, dim)) * 0.01
        missing_data_with_noise = mask * data + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()

        return cls.net_G(input_G)

    def impute(cls, data: Data):
        sample_G = cls.generate_sample(data.dataset_scaled, data.mask)
        data_imputed_scaled = data.dataset_scaled * data.mask + sample_G * (
            1 - data.mask
        )
        cls.metrics.data_imputed = data.scaler.inverse_transform(
            data_imputed_scaled.detach().numpy()
        )

    def _evaluate_impute(cls, data: Data, transpose=0):

        if transpose == 0:
            sample_G = cls.generate_sample(data.ref_dataset_scaled, data.ref_mask)
            data_imputed_scaled = data.ref_dataset_scaled * data.ref_mask + sample_G * (
                1 - data.ref_mask
            )
            cls.metrics.ref_data_imputed = data.scaler.inverse_transform(
                data_imputed_scaled.detach().numpy()
            )

            cls.metrics.ref_mean_imputed = data.scaler.inverse_transform(
                data.ref_mean_scaled.detach().numpy()
            )

            test_idx = torch.nonzero((data.mask - data.ref_mask) == 1)

            ref_imputed = np.empty((len(test_idx), 6))
            for i, id in enumerate(test_idx):
                ref_imputed[i] = np.array(
                    [
                        data.dataset_T[tuple(id)],
                        cls.metrics.ref_data_imputed[tuple(id)],
                        id[0],
                        id[1],
                        cls.metrics.ref_mean_imputed[tuple(id)],
                        data.ref_MF_imputed[tuple(id)],
                    ]
                )

            utils.create_csv(
                ref_imputed,
                f"{cls.hypers.output_folder}{cls.hypers.output_eval}",
                [
                    "original",
                    "imputed",
                    "sample",
                    "feature",
                    "mean_imputed",
                    "MF_imputed",
                ],
            )

        elif transpose == 1:
            sample_G = cls.generate_sample(data.ref_dataset_scaled_T, data.ref_mask_T)
            data_imputed_scaled = (
                data.ref_dataset_scaled_T * data.ref_mask_T
                + sample_G * (1 - data.ref_mask_T)
            )
            cls.metrics.ref_data_imputed_T = data.scaler.inverse_transform(
                data_imputed_scaled.T.detach().numpy()
            ).T

            cls.metrics.ref_mean_imputed_T = data.scaler.inverse_transform(
                data.ref_mean_scaled.detach().numpy()
            ).T

            test_idx = torch.nonzero((data.mask_T - data.ref_mask_T) == 1)

            ref_imputed = np.empty((len(test_idx), 6))
            for i, id in enumerate(test_idx):
                ref_imputed[i] = np.array(
                    [
                        data.dataset_T[tuple(id)],
                        cls.metrics.ref_data_imputed_T[tuple(id)],
                        id[0],
                        id[1],
                        cls.metrics.ref_mean_imputed_T[tuple(id)],
                        data.ref_MF_imputed_T[tuple(id)],
                    ]
                )

            utils.create_csv(
                ref_imputed,
                f"{cls.hypers.output_folder}{cls.hypers.output_eval}_transpose",
                [
                    "original",
                    "imputed",
                    "sample",
                    "feature",
                    "mean_imputed",
                    "MF_imputed",
                ],
            )

    def _update_G(cls, batch, mask, hint, Z, loss):
        loss_mse = nn.MSELoss(reduction="none")

        ones = torch.ones_like(batch)

        new_X = mask * batch + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)

        # print(batch, mask, ones.reshape(fake_Y.shape), fake_Y, loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask), (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask)).mean())
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1 - mask)
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G * mask).float(), (batch * mask).float())
        ).mean()

        loss_G = loss_G_entropy + cls.hypers.alpha * loss_G_mse

        cls.optimizer_G.zero_grad()
        loss_G.backward()
        cls.optimizer_G.step()

        return loss_G

    def _update_D(cls, batch, mask, hint, Z, loss):
        new_X = mask * batch + (1 - mask) * Z

        input_G = torch.cat((new_X, mask), 1).float()

        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)
        loss_D = (loss(fake_Y.float(), mask.float())).mean()
        cls.optimizer_D.zero_grad()
        loss_D.backward()
        cls.optimizer_D.step()

        return loss_D

    def train_ref(cls, data: Data, missing_header):

        dim = data.dataset_scaled.shape[1]
        train_size = data.dataset_scaled.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "Batch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:

            mb_idx = utils.sample_idx(train_size, cls.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone()
            mask_batch = data.mask[mb_idx].detach().clone()
            # hint_batch = data.hint[mb_idx].detach().clone()
            hint_batch = generate_hint(mask_batch, cls.hypers.hint_rate)
            ref_batch = data.ref_dataset_scaled[mb_idx].detach().clone()

            Z = torch.rand((cls.hypers.batch_size, dim)) * 0.01
            cls.metrics.loss_D[it] = cls._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            cls.metrics.loss_G[it] = cls._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = cls.generate_sample(batch, mask_batch)

            cls.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            cls.metrics.loss_MSE_test[it] = (
                loss_mse((1 - mask_batch) * ref_batch, (1 - mask_batch) * sample_G)
            ).mean() / (1 - mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D[it]: .3f}  loss G={cls.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train[it]): .4f}  rmse test={np.sqrt(cls.metrics.loss_MSE_test[it]): .3f}"
                pbar.clear()
                pbar.set_description(s)

            cls.metrics.cpu[it] = psutil.cpu_percent()
            cls.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        cls.impute(data)

        if cls.hypers.output_all == 1:
            utils.output(
                cls.metrics.data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D,
                cls.metrics.loss_G,
                cls.metrics.loss_MSE_train,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu,
                cls.metrics.ram,
                cls.metrics.ram_percentage,
                cls.hypers.override,
            )

    def evaluate(cls, data: Data, missing_header, transpose=0):

        if transpose == 0:
            dim = data.ref_dataset_scaled.shape[1]
            train_size = data.ref_dataset_scaled.shape[0]

        elif transpose == 1:
            dim = data.ref_dataset_scaled_T.shape[1]
            train_size = data.ref_dataset_scaled_T.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:
            mb_idx = utils.sample_idx(train_size, cls.hypers.batch_size)

            if transpose == 0:
                train_batch = data.ref_dataset_scaled[mb_idx].detach().clone()
                train_mask_batch = data.ref_mask[mb_idx].detach().clone()
                # train_hint_batch = data.ref_hint[mb_idx].detach().clone()
                train_hint_batch = generate_hint_paper_missingness(
                    train_mask_batch, cls.hypers.hint_rate, data.ref_missingness[mb_idx]
                )

                test_batch = data.dataset_scaled[mb_idx].detach().clone()
                test_mask_batch = data.mask[mb_idx].detach().clone()

                mean_batch = data.ref_mean_scaled[mb_idx].detach().clone()
            elif transpose == 1:
                train_batch = data.ref_dataset_scaled_T[mb_idx].detach().clone()
                train_mask_batch = data.ref_mask_T[mb_idx].detach().clone()
                # train_hint_batch = data.ref_hint[mb_idx].detach().clone()
                train_hint_batch = generate_hint_paper_missingness(
                    train_mask_batch,
                    cls.hypers.hint_rate,
                    data.ref_missingness[mb_idx],
                )

                test_batch = data.dataset_scaled_T[mb_idx].detach().clone()
                test_mask_batch = data.mask_T[mb_idx].detach().clone()

                mean_batch = data.ref_mean_scaled_T[mb_idx].detach().clone()

            Z = torch.rand((cls.hypers.batch_size, dim)) * 0.01
            cls.metrics.loss_D_evaluate[it] = cls._update_D(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            cls.metrics.loss_G_evaluate[it] = cls._update_G(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            sample_G = cls.generate_sample(train_batch, train_mask_batch)

            cls.metrics.loss_MSE_train_evaluate[it] = (
                loss_mse(train_mask_batch * train_batch, train_mask_batch * sample_G)
            ).mean()

            cls.metrics.loss_MSE_test[it] = (
                loss_mse(
                    (test_mask_batch - train_mask_batch) * test_batch,
                    (test_mask_batch - train_mask_batch) * sample_G,
                )
            ).mean() / (test_mask_batch - train_mask_batch).mean()

            mean_mse = (
                loss_mse(
                    (test_mask_batch - train_mask_batch) * test_batch,
                    (test_mask_batch - train_mask_batch) * mean_batch,
                )
            ).mean() / (test_mask_batch - train_mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D_evaluate[it]: .3f}  loss G={cls.metrics.loss_G_evaluate[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train_evaluate[it]): .4f}  rmse test={np.sqrt(cls.metrics.loss_MSE_test[it]): .3f} MEAN rmse test={np.sqrt(mean_mse): .3f}"
                pbar.clear()
                pbar.set_description(s)

            cls.metrics.cpu_evaluate[it] = psutil.cpu_percent()
            cls.metrics.ram_evaluate[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage_evaluate[it] = psutil.virtual_memory()[2]

        cls._evaluate_impute(data, transpose)

        if (cls.hypers.output_all == 1) & (transpose == 0):
            utils.output_eval(
                cls.metrics.ref_data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D_evaluate,
                cls.metrics.loss_G_evaluate,
                cls.metrics.loss_MSE_train_evaluate,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu_evaluate,
                cls.metrics.ram_evaluate,
                cls.metrics.ram_percentage_evaluate,
                cls.hypers.override,
            )

            final_mean_mse = []
            final_mean_mse.append(mean_mse.item())
            file_path = cls.hypers.output_folder + "final_mean_mse.csv"

            if cls.hypers.override == 1:
                df_final_mean_mse = pd.DataFrame(final_mean_mse)
                df_final_mean_mse.to_csv(file_path, index=False)

            else:
                if os.path.exists(file_path):
                    with open(file_path, "a") as myfile:
                        myfile.write(str(final_mean_mse[0]) + "\n")

                else:
                    df_final_mean_mse = pd.DataFrame(final_mean_mse)
                    df_final_mean_mse.to_csv(file_path, index=False)

        if transpose == 1:
            utils.output_eval_transpose(
                cls.metrics.ref_data_imputed_T,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D_evaluate,
                cls.metrics.loss_G_evaluate,
                cls.metrics.loss_MSE_train_evaluate,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu_evaluate,
                cls.metrics.ram_evaluate,
                cls.metrics.ram_percentage_evaluate,
                cls.hypers.override,
            )

            final_mean_mse = []
            final_mean_mse.append(mean_mse.item())
            file_path = cls.hypers.output_folder + "final_mean_mse_transpose.csv"

            if cls.hypers.override == 1:
                df_final_mean_mse = pd.DataFrame(final_mean_mse)
                df_final_mean_mse.to_csv(file_path, index=False)

            else:
                if os.path.exists(file_path):
                    with open(file_path, "a") as myfile:
                        myfile.write(str(final_mean_mse[0]) + "\n")

                else:
                    df_final_mean_mse = pd.DataFrame(final_mean_mse)
                    df_final_mean_mse.to_csv(file_path, index=False)

    def train(cls, data: Data, missing_header, transpose=1):

        for name, param in cls.net_D.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        for name, param in cls.net_G.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        cls.optimizer_D = torch.optim.Adam(cls.net_D.parameters(), lr=cls.hypers.lr_D)
        cls.optimizer_G = torch.optim.Adam(cls.net_G.parameters(), lr=cls.hypers.lr_G)

        if transpose == 0:
            dim = data.ref_dataset_scaled.shape[1]
            train_size = data.ref_dataset_scaled.shape[0]

        elif transpose == 1:
            dim = data.ref_dataset_scaled_T.shape[1]
            train_size = data.ref_dataset_scaled_T.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:

            mb_idx = utils.sample_idx(train_size, cls.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone()
            mask_batch = data.mask[mb_idx].detach().clone()
            # hint_batch = data.hint[mb_idx].detach().clone()
            hint_batch = generate_hint_paper_missingness(
                mask_batch, cls.hypers.hint_rate, data.missingness[mb_idx]
            )

            if transpose == 0:
                batch = data.dataset_scaled[mb_idx].detach().clone()
                mask_batch = data.mask[mb_idx].detach().clone()
                # train_hint_batch = data.ref_hint[mb_idx].detach().clone()
                hint_batch = generate_hint_paper_missingness(
                    mask_batch, cls.hypers.hint_rate, data.missingness[mb_idx]
                )
            elif transpose == 1:
                batch = data.dataset_scaled_T[mb_idx].detach().clone()
                mask_batch = data.mask_T[mb_idx].detach().clone()
                # train_hint_batch = data.ref_hint[mb_idx].detach().clone()
                hint_batch = generate_hint_paper_missingness(
                    mask_batch, cls.hypers.hint_rate, data.missingness[mb_idx]
                )

            Z = torch.rand((cls.hypers.batch_size, dim)) * 0.01
            cls.metrics.loss_D[it] = cls._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            cls.metrics.loss_G[it] = cls._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = cls.generate_sample(batch, mask_batch)

            cls.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D[it]: .3f}  loss G={cls.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train[it]): .4f}"
                pbar.clear()
                pbar.set_description(s)

            cls.metrics.cpu[it] = psutil.cpu_percent()
            cls.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        cls.impute(data)

        if cls.hypers.output_all == 1:
            utils.output(
                cls.metrics.data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D,
                cls.metrics.loss_G,
                cls.metrics.loss_MSE_train,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu,
                cls.metrics.ram,
                cls.metrics.ram_percentage,
                cls.hypers.override,
            )
