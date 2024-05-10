import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib import font_manager as fm
import math
import warnings
warnings.filterwarnings("ignore")

pi = np.pi


# 计算pie图所需参数
def count_x(x_data, pi=pi):
    x_data = np.array([0] + x_data)
    x_alpha = x_data * 2 * pi / np.sum(x_data)
    x_angle = [sum(x_alpha[:(i+1)]) for i in range(len(x_alpha))]
    k = [math.tan(i) for i in x_angle]
    x_angle[-1] = 2*pi
    return x_angle, k


# 绘制pie图边界线
def edge(x_angle, k, i, pi=pi, x0=0, y0=0, lw=2, lc='white', r=1, zorder=11):
    if x_angle[i] == 0.5 * pi:
        plt.plot([x0,x0], [y0,y0+r], lw=lw, color=lc, zorder=11)
    elif x_angle[i] == 1.5 * pi:
        plt.plot([x0,x0], [y0-r,y0], lw=lw, color=lc, zorder=11)
    else:
        I1 = math.cos(x_angle[i])/ abs(math.cos(x_angle[i]))
        I2 = math.sin(x_angle[i])/ abs(math.sin(x_angle[i]))
        x = I1 * r / ((k[i] ** 2) + 1) ** 0.5
        y = I2 * (r ** 2 - x ** 2) ** 0.5
        plt.plot([x0, x0+x], [y0, y0+y], lw=lw, color=lc, zorder=zorder)


# 绘制label
def plot_label(x_angle, k, label, i, x_data, pi=pi, exposure=False, x0=0, y0=0, fs=15, fc='black', rotation=True,
               threshold=0.05, r=1, show_value=False, show_percentage=False, r2=0.5, fs2=15, fc2='black', zorder=12):
    if not exposure:
        exposure = [0] * 999
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    k_text = math.tan(angle)

    if show_value and show_percentage:
        percentage = np.round(np.array([x_data[j] / sum(x_data) * 100 for j in range(len(x_data))]), 3)

    if angle == 0.5 * pi:
        plt.text(x0 + delta_x, y0 + (1 + threshold) * r + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=90 * rotation,
                 verticalalignment="bottom", horizontalalignment="center", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (delta_x), y0 + r2 * (r + delta_y), x_data[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (delta_x), y0 + r2 * (r + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == pi:
        plt.text(x0 - (1 + threshold) * r + delta_x, y0 + delta_y, label[i], fontsize=fs, color=fc, rotation=0,
                 verticalalignment="center", horizontalalignment="right", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 - r2 * (r + delta_x), y0 + r2 * (delta_y), x_data[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 - r2 * (r + delta_x), y0 + r2 * (delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == 1.5 * pi:
        plt.text(x0 + delta_x, y0 - (1 + threshold) * r + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=270 * rotation,
                 verticalalignment="top", horizontalalignment="center", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (delta_x), y0 - r2 * (r + delta_y), x_data[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (delta_x), y0 - r2 * (r + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle < 0.5 * pi:
        I1 = math.cos(angle) / abs(math.cos(angle))
        I2 = math.sin(angle) / abs(math.sin(angle))
        x = I1 * (r / ((k_text ** 2) + 1) ** 0.5)
        y = I2 * (r ** 2 - x ** 2) ** 0.5

        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), x_data[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r
        y += threshold * math.sin(angle) * r
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * angle * 180 / pi,
                 verticalalignment="baseline", horizontalalignment="left", zorder=zorder)

    elif angle > 1.5 * pi:
        I1 = math.cos(angle) / abs(math.cos(angle))
        I2 = math.sin(angle) / abs(math.sin(angle))
        x = I1 * (r / ((k_text ** 2) + 1) ** 0.5)
        y = I2 * (r ** 2 - x ** 2) ** 0.5

        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), x_data[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r
        y += threshold * math.sin(angle) * r
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * angle * 180 / pi,
                 verticalalignment="top", horizontalalignment="left", zorder=zorder)

    elif pi < angle < 1.5 * pi:
        I1 = math.cos(angle) / abs(math.cos(angle))
        I2 = math.sin(angle) / abs(math.sin(angle))
        x = I1 * (r / ((k_text ** 2) + 1) ** 0.5)
        y = I2 * (r ** 2 - x ** 2) ** 0.5

        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), x_data[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r
        y += threshold * math.sin(angle) * r
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * (180 + angle * 180 / pi),
                 verticalalignment="top", horizontalalignment="right", zorder=zorder)

    else:
        I1 = math.cos(angle) / abs(math.cos(angle))
        I2 = math.sin(angle) / abs(math.sin(angle))
        x = I1 * (r / ((k_text ** 2) + 1) ** 0.5)
        y = I2 * (r ** 2 - x ** 2) ** 0.5

        if show_value and (not show_percentage):
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), x_data[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2 * (x + delta_x), y0 + r2 * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r
        y += threshold * math.sin(angle) * r
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * (180 + angle * 180 / pi),
                 verticalalignment="baseline", horizontalalignment="right", zorder=zorder)

# 绘制等高线
def contour(r, figsize=False, xrange=False, epsilon=0.001, linestyle='auto', color='grey', zorder=1):
    if not figsize and linestyle == 'auto':
        figsize = 6
        print('Warning: missing the parameter \'figsize\' for auto linestyle!')
    if not xrange and linestyle == 'auto':
        xrange = r
        print('Warning: missing the parameter \'xrange\' for auto linestyle!')

    if linestyle == 'auto':  # 自动计算线长，使得虚线间隔合理
        coef = figsize * r / xrange / 10
        linestyle = (98 * coef, (191 * coef, 20 * coef))

    epsilon = 0.001
    x1 = np.arange(-r, r, epsilon)
    x2 = np.arange(-r, r, epsilon)
    y1 = (r ** 2 - x1 ** 2) ** 0.5
    y2 = -(r ** 2 - x2 ** 2) ** 0.5
    plt.plot(x1, y1, linestyle=linestyle, color=color, zorder=zorder)
    plt.plot(x2, y2, linestyle=linestyle, color=color, zorder=zorder)


# pie 主体
def pi_1_1(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(0, x2, epsilon)
    x_plot_b = np.arange(x2, x1, epsilon)

    y_plot_f_u = k[i] * x_plot_f
    y_plot_f_d = k[i - 1] * x_plot_f
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)

def pi_1_2(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])

    x_plot_b = np.arange(0, x1, epsilon)
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    plt.fill_between(x_plot_b + x0 + delta_x, y_plot_b_u + y0 + delta_y, y_plot_b_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = -r / ((k[i] ** 2) + 1) ** 0.5

    x_plot_f = np.arange(x2, 0, epsilon)
    x_plot_b = np.arange(0, x1, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = k[i] * x_plot_f
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = -r / ((k[i] ** 2) + 1) ** 0.5

    x_plot_f = np.arange(-r, x2, epsilon)
    x_plot_m = np.arange(x2, 0, epsilon)
    x_plot_b = np.arange(0, x1, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = (r ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = k[i] * x_plot_m
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])

    x_plot_f = np.arange(-r, 0, epsilon)
    x_plot_b = np.arange(0, x1, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    y1 = r * math.sin(x_angle[i-1])
    y2 = r * math.sin(x_angle[i])

    y_plot_0 = np.arange(-r, r, epsilon)
    y_plot_1 = np.arange(-r, y2, epsilon)
    y_plot_2 = np.arange(y2, 0, epsilon)
    y_plot_3 = np.arange(0, y1, epsilon)
    y_plot_4 = np.arange(y1, r, epsilon)

    y_plot_0 = np.append(y_plot_1, y_plot_2)
    y_plot_0 = np.append(y_plot_0, y_plot_3)
    y_plot_0 = np.append(y_plot_0, y_plot_4)

    x_plot_l = - (r ** 2 - y_plot_0 ** 2) ** 0.5
    x_plot_1 = (r ** 2 - y_plot_1 ** 2) ** 0.5
    x_plot_2 = y_plot_2 / k[i]
    x_plot_3 = y_plot_3 / k[i-1]
    x_plot_4 = (r ** 2 - y_plot_4 ** 2) ** 0.5

    x_plot_r = np.append(x_plot_1, x_plot_2)
    x_plot_r = np.append(x_plot_r, x_plot_3)
    x_plot_r = np.append(x_plot_r, x_plot_4)

    plt.fill_betweenx(y_plot_0 + y0 + delta_y, x_plot_l + x0 + delta_x, x_plot_r + x0 + delta_x, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x2 = -r / ((k[i] ** 2) + 1) ** 0.5

    x_plot_f = np.arange(x2, epsilon, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = k[i] * x_plot_f

    plt.fill_between(x_plot_f + x0 + delta_x, y_plot_f_u + y0 + delta_y, y_plot_f_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x2 = -r / ((k[i] ** 2) + 1) ** 0.5

    x_plot_f = np.arange(-r, x2, epsilon)
    x_plot_b = np.arange(x2, 0, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = (r ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x_plot_f = np.arange(-r, epsilon, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5

    plt.fill_between(x_plot_f + x0 + delta_x, y_plot_f_u + y0 + delta_y, y_plot_f_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(-r, 0, epsilon)
    x_plot_b = np.arange(0, x2, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_3_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(x2, x1, epsilon)
    x_plot_b = np.arange(x1, 0, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = k[i] * x_plot_f
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_3_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_1 = np.arange(-r, x1, epsilon)
    x_plot_2 = np.arange(x1, 0, epsilon)
    x_plot_3 = np.arange(-r, x2, epsilon)
    x_plot_4 = np.arange(x2, 0, epsilon)

    if len(x_plot_1) == 0:
        x_plot_1 = np.array([-r])
    elif len(x_plot_2) == 0:
        x_plot_2 = np.array([0])
    if len(x_plot_3) == 0:
        x_plot_3 = np.array([-r])
    elif len(x_plot_4) == 0:
        x_plot_4 = np.array([0])

    if (len(x_plot_1) + len(x_plot_2) > len(x_plot_3) + len(x_plot_4)):
        x_plot_1 = x_plot_1[:-1]
    elif (len(x_plot_1) + len(x_plot_2) < len(x_plot_3) + len(x_plot_4)):
        x_plot_3 = x_plot_3[:-1]

    x_plot = np.append(x_plot_1, x_plot_2)

    y_plot_1 = (r ** 2 - x_plot_1 ** 2) ** 0.5
    y_plot_2 = k[i-1] * x_plot_2
    y_plot_3 = - (r ** 2 - x_plot_3 ** 2) ** 0.5
    y_plot_4 = k[i] * x_plot_4

    y_plot_u = np.append(y_plot_1, y_plot_2)
    y_plot_d = np.append(y_plot_3, y_plot_4)

    # return print(len(x_plot_1), len(x_plot_2), len(x_plot_3), len(x_plot_4))

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)

def pi_3_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])

    x_plot_f = np.arange(-r, x1, epsilon)
    x_plot_b = np.arange(x1, 0, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_3_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(-r, x1, epsilon)
    x_plot_m = np.arange(x1, 0, epsilon)
    x_plot_b = np.arange(0, x2, epsilon)

    y_plot_f_u = (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5

    y_plot_m_u = k[i - 1] * x_plot_m
    y_plot_m_d = - (r ** 2 - x_plot_m ** 2) ** 0.5

    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_4_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2 + epsilon, epsilon)
    x_plot_b = np.arange(x2, epsilon, epsilon)

    y_plot_f_u = k[i - 1] * x_plot_f
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_4_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])

    x_plot_f = np.arange(x1, epsilon, epsilon)

    y_plot_f_u = k[i - 1] * x_plot_f
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5

    plt.fill_between(x_plot_f + x0 + delta_x, y_plot_f_u + y0 + delta_y, y_plot_f_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_4_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, epsilon, epsilon)
    x_plot_b = np.arange(0, x2 + epsilon, epsilon)

    y_plot_f_u = k[i - 1] * x_plot_f
    y_plot_f_d = - (r ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_5_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x2 = r * math.cos(x_angle[i])

    x_plot_b = np.arange(0, x2 + epsilon, epsilon)

    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    plt.fill_between(x_plot_b + x0 + delta_x, y_plot_b_u + y0 + delta_y, y_plot_b_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_6_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r * math.cos(angle)
    delta_y = exposure[i] * r * math.sin(angle)
    x1 = r * math.cos(x_angle[i-1])
    x2 = r * math.cos(x_angle[i])

    x_plot_f = np.arange(0, x1 + epsilon, epsilon)
    x_plot_b = np.arange(x1, x2 + epsilon, epsilon)

    y_plot_f_u = k[i] * x_plot_f
    y_plot_f_d = k[i - 1] * x_plot_f
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_b)
    y_plot_u = np.append(y_plot_f_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y, color=color_list[i], alpha=alpha[i], zorder=zorder)


# 主程序
def Control_center(x_angle, k, i, pi=pi, color_list=False, exposure=False, epsilon=0.001, r=1, alpha=False, x0=0, y0=0, zorder=10):
    if not color_list:
        color_list = ['']+[ 'lightpink', 'lightskyblue', 'wheat', 'mediumpurple']*200
    if not exposure:
        exposure = [0] * 999
    if not alpha:
        alpha = [1] * 999

    # 1-#
    if x_angle[i - 1] < 0.5 * pi:
        if x_angle[i] < 0.5 * pi:  # 1-1
            pi_1_1(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif x_angle[i] == 0.5 * pi:  # 1-2
            pi_1_2(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif 0.5 * pi < x_angle[i] < pi:  # 1-3
            pi_1_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif pi <= x_angle[i] < 1.5 * pi:  # 1-4
            pi_1_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif x_angle[i] == 1.5 * pi:  # 1-5
            pi_1_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif 1.5 * pi < x_angle[i] <= 2 * pi:  # 1-6
            pi_1_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)

    # 2-#
    elif x_angle[i - 1] == 0.5 * pi:
        if 0.5 * pi < x_angle[i] < pi:  # 2-3
            pi_2_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif pi <= x_angle[i] < 1.5 * pi:  # 2-4
            pi_2_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif x_angle[i] == 1.5 * pi:  # 2-5
            pi_2_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif 1.5 * pi < x_angle[i] <= 2 * pi:  # 2-6
            pi_2_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)

    # 3-#
    elif 0.5 * pi < x_angle[i - 1] < pi:
        if 0.5 * pi < x_angle[i] < pi:  # 3-3
            pi_3_3(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif pi <= x_angle[i] < 1.5 * pi:  # 3-4
            pi_3_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif x_angle[i] == 1.5 * pi:  # 3-5
            pi_3_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif 1.5 * pi < x_angle[i] <= 2 * pi:  # 3-6
            pi_3_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)

    # 4-#
    elif pi <= x_angle[i - 1] < 1.5 * pi:
        if pi <= x_angle[i] < 1.5 * pi:  # 4-4
            pi_4_4(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif x_angle[i] == 1.5 * pi:  # 4-5
            pi_4_5(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)
        elif 1.5 * pi < x_angle[i] <= 2 * pi:  # 4-6
            pi_4_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)

    # 5-#
    elif x_angle[i - 1] == 1.5 * pi:
        if 1.5 * pi < x_angle[i] <= 2 * pi:  # 5-6
            pi_5_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)

    # 6-#
    else:  # 6-6
        pi_6_6(x_angle, k, i, exposure, epsilon, color_list, r, alpha, x0, y0, zorder)