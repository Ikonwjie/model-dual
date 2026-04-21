# -*- coding: utf-8 -*-
"""评估指标"""

import numpy as np


def RSE(pred, true):
    """相对平方误差"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    """相关系数"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """平均绝对误差"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """均方误差"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """均方根误差"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """平均绝对百分比误差"""
    return np.mean(np.abs((pred - true) / (true + 1e-8)))


def MSPE(pred, true):
    """均方百分比误差"""
    return np.mean(np.square((pred - true) / (true + 1e-8)))


def metric(pred, true):
    """计算所有指标"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
