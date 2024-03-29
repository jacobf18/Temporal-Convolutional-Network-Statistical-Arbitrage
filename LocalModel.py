import torch, h5py
import numpy as np
from scipy.io import loadmat
from torch.nn.utils import weight_norm

import torch.nn as nn
import torch.optim as optim
import numpy as np

# import matplotlib
from torch.autograd import Variable


import itertools
import torch.nn.functional as F


from DeepGLO.data_loader import *

use_cuda = True  #### Assuming you have a GPU ######

from DeepGLO.utilities import *
from DeepGLO.time import *
from DeepGLO.metrics import *
import random
import pickle

np.random.seed(111)
torch.cuda.manual_seed(111)
torch.manual_seed(111)
random.seed(111)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.1,
        init=True,
    ):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)

            self.conv1.weight[:, 0, :] += (
                1.0 / self.kernel_size
            )  ###new initialization scheme
            self.conv2.weight += 1.0 / self.kernel_size  ###new initialization scheme

            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlock_last(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        init=True,
    ):
        super(TemporalBlock_last, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)

            self.conv1.weight[:, 0, :] += (
                1.0 / self.kernel_size
            )  ###new initialization scheme
            self.conv2.weight += 1.0 / self.kernel_size  ###new initialization scheme

            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, init=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.dropout = dropout
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    TemporalBlock_last(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dropout=dropout,
                        init=init,
                    )
                ]
            else:
                layers += [
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dropout=dropout,
                        init=init,
                    )
                ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LocalModel(object):
    def __init__(
        self,
        Ymat,
        num_inputs=1,
        num_channels=[32, 32, 32, 32, 32, 1],
        kernel_size=7,
        dropout=0.2,
        vbsize=300,
        hbsize=128,
        num_epochs=100,
        lr=0.0005,
        val_len=10,
        test=True,
        end_index=120,
        normalize=False,
        start_date="2016-1-1",
        freq="H",
        covariates=None,
        use_time=False,
        dti=None,
        Ycov=None,
    ):
        """
        Arguments:
        Ymat: input time-series n*T
        num_inputs: always set to 1
        num_channels: list containing channel progression of temporal comvolution network
        kernel_size: kernel size of temporal convolution filters
        dropout: dropout rate for each layer
        vbsize: vertical batch size
        hbsize: horizontal batch size
        num_epochs: max. number of epochs
        lr: learning rate
        val_len: validation length
        test: always set to True
        end_index: no data is touched fro training or validation beyond end_index
        normalize: normalize dataset before training or not
        start_data: start data in YYYY-MM-DD format (give a random date if unknown)
        freq: "H" hourly, "D": daily and for rest see here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        covariates: global covariates common for all time series r*T, where r is the number of covariates
        Ycov: per time-series covariates n*l*T, l such covariates per time-series
        use_time: if false, default trime-covriates are not used
        dti: date time object can be explicitly supplied here, leave None if default options are to be used
        """
        self.start_date = start_date
        if use_time:
            self.time = TimeCovariates(
                start_date=start_date, freq=freq, normalized=True, num_ts=Ymat.shape[1]
            )
            if dti is not None:
                self.time.dti = dti
            time_covariates = self.time.get_covariates()
            if covariates is None:
                self.covariates = time_covariates
            else:
                self.covariates = np.vstack([time_covariates, covariates])
        else:
            self.covariates = covariates
        self.Ycov = Ycov
        self.freq = freq
        self.vbsize = vbsize
        self.hbsize = hbsize
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.num_epochs = num_epochs
        self.lr = lr
        self.val_len = val_len
        self.Ymat = Ymat
        self.test = test
        self.end_index = end_index
        self.normalize = normalize
        self.kernel_size = kernel_size
        if normalize:
            Y = Ymat
            m = np.mean(Y[:, 0 : self.end_index], axis=1)
            s = np.std(Y[:, 0 : self.end_index], axis=1)
            # s[s == 0] = 1.0
            s += 1.0
            Y = (Y - m[:, None]) / s[:, None]
            mini = np.abs(np.min(Y))
            self.Ymat = Y + mini

            self.m = m
            self.s = s
            self.mini = mini

        if self.Ycov is not None:
            self.num_inputs += self.Ycov.shape[1]
        if self.covariates is not None:
            self.num_inputs += self.covariates.shape[0]

        self.seq = TemporalConvNet(
            num_inputs=self.num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            init=True,
        )

        self.seq = self.seq.float()

        self.D = data_loader(
            Ymat=self.Ymat,
            vbsize=vbsize,
            hbsize=hbsize,
            end_index=end_index,
            val_len=val_len,
            covariates=self.covariates,
            Ycov=self.Ycov,
        )
        self.val_len = val_len
        if use_cuda:
            self.seq = self.seq.cuda()

    def __loss__(self, out, target, dic=None):
        criterion = nn.L1Loss()
        return criterion(out, target) / torch.abs(target.data).mean()

    def __prediction__(self, data):
        dic = None
        out = self.seq(data)
        return out, dic

    def train_model(self, early_stop=False, tenacity=10):
        """
        early_stop: set true for using early stop
        tenacity: patience for early_stop
        """
        print("Training Local Model(Tconv)")
        if use_cuda:
            self.seq = self.seq.cuda()
        optimizer = optim.Adam(params=self.seq.parameters(), lr=self.lr)
        iter_count = 0
        loss_all = []
        loss_test_all = []
        vae = float("inf")
        scount = 0
        while self.D.epoch < self.num_epochs:
            last_epoch = self.D.epoch
            inp, out_target, _, _ = self.D.next_batch()
            if self.test:
                inp_test, out_target_test, _, _ = self.D.supply_test()
            current_epoch = self.D.epoch
            if use_cuda:
                inp = inp.cuda()
                out_target = out_target.cuda()
            inp = Variable(inp)
            out_target = Variable(out_target)
            optimizer.zero_grad()
            out, dic = self.__prediction__(inp)
            loss = self.__loss__(out, out_target, dic)
            iter_count = iter_count + 1
            for p in self.seq.parameters():
                p.requires_grad = True
            loss.backward()
            for p in self.seq.parameters():
                p.grad.data.clamp_(max=1e5, min=-1e5)
            optimizer.step()

            loss_all = loss_all + [loss.cpu().item()]
            if self.test:
                if use_cuda:
                    inp_test = inp_test.cuda()
                    out_target_test = out_target_test.cuda()
                inp_test = Variable(inp_test)
                out_target_test = Variable(out_target_test)
                out_test, dic = self.__prediction__(inp_test)
                losst = self.__loss__(out_test, out_target_test, dic)
            loss_test_all = loss_test_all + [losst.cpu().item()]

            if current_epoch > last_epoch:
                ve = loss_test_all[-1]
                print("Entering Epoch# ", current_epoch)
                print("Train Loss:", np.mean(loss_all))
                print("Validation Loss:", ve)
                if ve <= vae:
                    vae = ve
                    scount = 0
                    # self.saved_seq = TemporalConvNet(
                    #     num_inputs=self.seq.num_inputs,
                    #     num_channels=self.seq.num_channels,
                    #     kernel_size=self.seq.kernel_size,
                    #     dropout=self.seq.dropout,
                    # )
                    # self.saved_seq.load_state_dict(self.seq.state_dict())
                    self.saved_seq = pickle.loads(pickle.dumps(self.seq))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        self.seq = self.saved_seq
                        if use_cuda:
                            self.seq = self.seq.cuda()

                        break

    def convert_to_input(self, data, cuda=True):
        n, m = data.shape
        inp = torch.from_numpy(data).view(1, n, m)
        inp = inp.transpose(0, 1).float()

        if cuda:
            inp = inp.cuda()

        return inp

    def convert_covariates(self, data, covs, cuda=True):
        nd, td = data.shape
        rcovs = np.repeat(
            covs.reshape(1, covs.shape[0], covs.shape[1]), repeats=nd, axis=0
        )
        rcovs = torch.from_numpy(rcovs).float()
        if cuda:
            rcovs = rcovs.cuda()
        return rcovs

    def convert_ycovs(self, data, ycovs, cuda=True):
        nd, td = data.shape
        ycovs = torch.from_numpy(ycovs).float()
        if cuda:
            ycovs = ycovs.cuda()
        return ycovs

    def convert_from_output(self, T):
        out = T.view(T.size(0), T.size(2))
        return np.array(out.cpu().detach())

    def predict_future_batch(
        self, data, covariates=None, ycovs=None, future=10, cpu=False
    ):
        if cpu:
            self.seq = self.seq.cpu()
        else:
            self.seq = self.seq.cuda()
        inp = self.convert_to_input(data)
        if covariates is not None:
            cov = self.convert_covariates(data, covariates)
            inp = torch.cat((inp, cov[:, :, 0 : inp.size(2)]), 1)
        if ycovs is not None:
            ycovs = self.convert_ycovs(data, ycovs)
            inp = torch.cat((inp, ycovs[:, :, 0 : inp.size(2)]), 1)
        if cpu:
            inp = inp.cpu()
            cov = cov.cpu()
            ycovs = ycovs.cpu()
        out, dic = self.__prediction__(inp)
        ci = inp.size(2)
        output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
        if covariates is not None:
            output = torch.cat(
                (output, cov[:, :, ci].view(cov.size(0), cov.size(1), 1)), 1
            )
        if ycovs is not None:
            output = torch.cat(
                (output, ycovs[:, :, ci].view(ycovs.size(0), ycovs.size(1), 1)), 1
            )
        out = torch.cat((inp, output), dim=2)
        for i in range(future - 1):
            inp = out
            out, dic = self.__prediction__(inp)
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            ci += 1
            if covariates is not None:
                output = torch.cat(
                    (output, cov[:, :, ci].view(cov.size(0), cov.size(1), 1)), 1
                )
            if ycovs is not None:
                output = torch.cat(
                    (output, ycovs[:, :, ci].view(ycovs.size(0), ycovs.size(1), 1)), 1
                )
            out = torch.cat((inp, output), dim=2)
        out = out[:, 0, :].view(out.size(0), 1, out.size(2))
        out = out.cuda()
        y = self.convert_from_output(out)
        self.seq = self.seq.cuda()
        return y

    def predict_future(
        self,
        data_in,
        covariates=None,
        ycovs=None,
        future=10,
        cpu=False,
        bsize=40,
        normalize=False,
    ):
        """
        data_in: input past data in same format of Ymat
        covariates: input past covariates
        ycovs: input past individual covariates
        future: number of time-points to predict
        cpu: if true then gpu is not used
        bsize: batch size for processing (determine according to gopu memory limits)
        normalize: should be set according to the normalization used in the class initialization
        """
        if normalize:
            data = (data_in - self.m[:, None]) / self.s[:, None]
            data += self.mini

        else:
            data = data_in

        n, T = data.shape

        I = list(np.arange(0, n, bsize))
        I.append(n)
        bdata = data[range(I[0], I[1]), :]
        if ycovs is not None:
            out = self.predict_future_batch(
                bdata, covariates, ycovs[range(I[0], I[1]), :, :], future, cpu
            )
        else:
            out = self.predict_future_batch(bdata, covariates, None, future, cpu)

        for i in range(1, len(I) - 1):
            bdata = data[range(I[i], I[i + 1]), :]
            self.seq = self.seq.cuda()
            if ycovs is not None:
                temp = self.predict_future_batch(
                    bdata, covariates, ycovs[range(I[i], I[i + 1]), :, :], future, cpu
                )
            else:
                temp = self.predict_future_batch(bdata, covariates, None, future, cpu)
            out = np.vstack([out, temp])

        if normalize:
            temp = (out - self.mini) * self.s[:, None] + self.m[:, None]
            out = temp

        return out

    def rolling_validation(self, Ymat, tau=24, n=7, bsize=90, cpu=False, alpha=0.3):
        last_step = Ymat.shape[1] - tau * n
        rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels) - 1)
        self.seq = self.seq.eval()
        if self.covariates is not None:
            covs = self.covariates[:, last_step - rg : last_step + tau]
        else:
            covs = None
        if self.Ycov is not None:
            ycovs = self.Ycov[:, :, last_step - rg : last_step + tau]
        else:
            ycovs = None
        data_in = Ymat[:, last_step - rg : last_step]
        out = self.predict_future(
            data_in,
            covariates=covs,
            ycovs=ycovs,
            future=tau,
            cpu=cpu,
            bsize=bsize,
            normalize=self.normalize,
        )
        predicted_values = []
        actual_values = []
        S = out[:, -tau::]
        predicted_values += [S]
        R = Ymat[:, last_step : last_step + tau]
        actual_values += [R]
        print("Current window wape: " + str(wape(S, R)))

        for i in range(n - 1):
            last_step += tau
            rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels) - 1)
            if self.covariates is not None:
                covs = self.covariates[:, last_step - rg : last_step + tau]
            else:
                covs = None
            if self.Ycov is not None:
                ycovs = self.Ycov[:, :, last_step - rg : last_step + tau]
            else:
                ycovs = None
            data_in = Ymat[:, last_step - rg : last_step]
            out = self.predict_future(
                data_in,
                covariates=covs,
                ycovs=ycovs,
                future=tau,
                cpu=cpu,
                bsize=bsize,
                normalize=self.normalize,
            )
            S = out[:, -tau::]
            predicted_values += [S]
            R = Ymat[:, last_step : last_step + tau]
            actual_values += [R]
            print("Current window wape: " + str(wape(S, R)))

        predicted = np.hstack(predicted_values)
        actual = np.hstack(actual_values)

        dic = {}
        dic["wape"] = wape(predicted, actual)
        dic["mape"] = mape(predicted, actual)
        dic["smape"] = smape(predicted, actual)
        dic["mae"] = np.abs(predicted - actual).mean()
        dic["rmse"] = np.sqrt(((predicted - actual) ** 2).mean())
        dic["nrmse"] = dic["rmse"] / np.sqrt(((actual) ** 2).mean())

        baseline = Ymat[:, Ymat.shape[1] - n * tau - tau : Ymat.shape[1] - tau]
        dic["baseline_wape"] = wape(baseline, actual)
        dic["baseline_mape"] = mape(baseline, actual)
        dic["baseline_smape"] = smape(baseline, actual)

        return dic