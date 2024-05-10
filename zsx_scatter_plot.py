#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def bar_scatter(data, xs, ys, marker=False, s=False, c=False, yeld=False,
                lw=False, lc=False, ls=False, k=False):
    default = {'marker': ['D'], 's': [80], 'c': ['red'], 'yeld': [1.5],
               'lw': [3], 'lc': ['gray'], 'ls': ['-'], 'k': [0.2]}
    for para in ['marker', 's', 'c', 'yeld', 'lw', 'lc', 'ls', 'k']:
        if not exec(para):
            code = para + ' = default[\'' + para + '\'] * data.shape[0]'
            exec(code)

    # 自动计算xy的边界1
    comput_xmax = []
    comput_xmin = []
    comput_ymax = []
    comput_ymin = []
    mult = ys / xs

    for i in range(int(data.shape[1] / 2)):
        da = data.iloc[:, 2 * i:2 * (i + 1)]
        da = da.loc[da.iloc[:, 0] != 'Na'].astype(float)

        u_x = np.mean(da.iloc[:, 0])
        d_x = np.std(da.iloc[:, 0]) * yeld[i]
        u_y = np.mean(da.iloc[:, 1])
        d_y = np.std(da.iloc[:, 1]) * yeld[i]

        comput_xmax += [max(da.iloc[:, 0])] + [u_x+d_x]
        comput_xmin += [min(da.iloc[:, 0])] + [u_x-d_x]
        comput_ymax += [max(da.iloc[:, 1])] + [u_y+d_y]
        comput_ymin += [min(da.iloc[:, 1])] + [u_y-d_y]

    xmax = max(comput_xmax)
    xmin = min(comput_xmin)
    ymax = max(comput_ymax)
    ymin = min(comput_ymin)
    xd = xmax - xmin
    yd = ymax - ymin
    xylim = [xmin - 0.15 * xd, xmax + 0.15 * xd, ymin - 0.15 * yd, ymax + 0.15 * yd]

    for i in range(int(data.shape[1] / 2)):
        da = data.iloc[:, 2 * i:2 * (i + 1)]
        da = da.loc[da.iloc[:, 0] != 'Na'].astype(float)
        lab = data.columns[2 * i]

        # 计算数据
        u_x = np.mean(da.iloc[:, 0])
        d_x = np.std(da.iloc[:, 0]) * yeld[i]
        u_y = np.mean(da.iloc[:, 1])
        d_y = np.std(da.iloc[:, 1]) * yeld[i]


        # 误差主线
        plt.plot([u_x - d_x, u_x + d_x], [u_y, u_y], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)
        plt.plot([u_x, u_x], [u_y - d_y, u_y + d_y], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)

        # 横向主线对应的小线
        plt.plot([u_x - d_x, u_x - d_x], [u_y - yd * k[i], u_y + yd * k[i]], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)
        plt.plot([u_x + d_x, u_x + d_x], [u_y - yd * k[i], u_y + yd * k[i]], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)

        # 纵向主线对应的小线
        plt.plot([u_x - mult * xd * k[i], u_x + mult * xd * k[i]], [u_y + d_y, u_y + d_y], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)
        plt.plot([u_x - mult * xd * k[i], u_x + mult * xd * k[i]], [u_y - d_y, u_y - d_y], lw=lw[i], c=lc[i], ls=ls[i], zorder=0)

        # 散点
        plt.scatter(u_x, u_y, s=s[i], marker=marker[i], c=c[i], zorder=1, label=lab)

    return xylim, plt