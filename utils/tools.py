# -*- coding: utf-8 -*-
"""工具函数"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from utils.metrics import MAPE

plt.switch_backend("agg")


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    """调整学习率"""
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate
            if epoch < 3
            else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            if accelerator is not None:
                accelerator.print("Updating learning rate to {}".format(lr))
            else:
                print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    """早停类"""
    def __init__(
        self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True
    ):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            else:
                self.accelerator.print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            else:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + "/" + "checkpoint")
        else:
            torch.save(model.state_dict(), path + "/" + "checkpoint")
        self.val_loss_min = val_loss


class dotdict(dict):
    """支持点符号访问的字典"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    """标准化器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def del_files(dir_path):
    """删除目录"""
    shutil.rmtree(dir_path)


def vali(
    args,
    accelerator,
    model,
    vali_data,
    vali_loader,
    criterion,
    mae_metric,
    epoch,
    mode="vali",
):
    """验证函数（适配BALM_MedualTime模型 - 时间序列预测任务）"""
    total_loss = []
    total_mae_loss = []
    total_mape_loss = []
    preds = []
    trues = []
    model.eval()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(vali_loader)
        ):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            
            # 模型前向传播（评估模式）- 输出 [batch, pred_len, n_vars]
            outputs = model(batch_x, mode='eval')
            
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == "MS" else 0
            
            # 提取预测结果和目标
            if args.features == "MS":
                outputs = outputs[:, :, f_dim:]
            
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            preds.append(pred)
            trues.append(true)

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()

            mape_loss = MAPE(pred_np, true_np)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_mape_loss.append(mape_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_mape_loss = np.average(total_mape_loss)

    preds = [
        pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred for pred in preds
    ]
    trues = [
        true.cpu().numpy() if isinstance(true, torch.Tensor) else true for true in trues
    ]
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)

    model.train()
    return total_loss, total_mae_loss, total_mape_loss, preds, trues


def vali_forecast(
    args,
    accelerator,
    model,
    vali_data,
    vali_loader,
    criterion,
    mae_metric,
    epoch,
    mode="vali",
):
    """验证函数（适配预测任务）"""
    total_loss = []
    total_mae_loss = []
    preds = []
    trues = []
    model.eval()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(vali_loader)
        ):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            
            # 模型前向传播
            outputs = model(batch_x, mode='eval')
            
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach()
            true = batch_y.detach()

            preds.append(pred)
            trues.append(true)

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    preds = torch.cat(preds, 0).cpu().numpy()
    trues = torch.cat(trues, 0).cpu().numpy()

    model.train()
    return total_loss, total_mae_loss, preds, trues
