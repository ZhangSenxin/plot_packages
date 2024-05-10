#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math


# main
def ring_plot(x_data_, r1=1, r2=0.8, color_list=None, exposure=None, epsilon=0.001,
              alpha=None, x0=0, y0=0, zorder_ring=10, angle_fold_change=1, angle_start_change=0,
              edge_open=False, lw=2, lc='white', zorder_edge=11, pie=None,
              label=None, fs=15, fc='black', rotation=True, threshold=0.12,
              show_value=False, show_percentage=False, number_threshold=0.2, fs2=15, fc2='black', zorder_label=12,
              r_list=None, contour_ls='auto', figsize=None, xrange=None, contour_lc='grey', zorder_contour=1):
    """
    Draw a ring plot in a Cartesian coordinate system.
    :param x_data_: listlike of floats, each corresponding to the central angle.
    :param r1: float, outside radius of the ring.
    :param r2: float, inside radius of the ring.
    :param color_list: listlike of colors, each corresponding to 'x_data_', default 4 colors.
    :param exposure: listlike of floats, each corresponding to 'x_data_', distance out of the center of the ring.
    :param epsilon: Tthe smaller the value, the higher the accuracy of the drawing, default 0.001.
    :param alpha: listlike of floats, each corresponding to 'color_list', transparency.
    :param x0: float, center x-coordinate of the ring, default 0.
    :param y0: float, center y-coordinate of the ring, default 0.
    :param zorder_ring: zorder of the ring, default 10.
    :param angle_fold_change: The overall zoomed-out magnification of the "ring",
                              which is often used to create a ring plot with gaps.
    :param angle_start_change: Drawing starting radian, default starting from radian 0 (unit vector is (1, 0)).
    :param edge_open: drawing edge or not, default False.
    :param lw: linewidth of the edge, default 2.
    :param lc: linecolor of the edge, default 'white'.
    :param zorder_edge: zorder of the edge, default 11.
    :param pie: value of pie.
    :param label: listlike of strings, each corresponding to 'x_data_', default None.
    :param fs: label fontsize, default 15.
    :param fc: label fontcolor, default 'black'.
    :param rotation: label text rotation. True, False, and 'center' is allowed, default True.
    :param threshold: threshold to control which label will be displayed
                      if the correspondiong radian exceeds the threshold, default 0.12 (pie)
    :param show_value: show the value or not, only effective when show_value=True, default False.
    :param show_percentage: show value in percentage or true value, default False.
    :param number_threshold: threshold to control which value will be displayed
                             if the correspondiong radian exceeds the threshold,
                             only effective when show_value=True, default 0.12 (pie)
    :param fs: value fontsize, default 15.
    :param fc: value fontcolor, default 'black'.
    :param zorder_label: zorder of the labels and values, default 12.
    :param r_list: contour lines‘ radius.
    :param contour_ls: contour linestyle, default 'auto'.
    :param figsize: max(figsize), default None.
    :param xrange: xrange for contour, default None
    :param contour_lc: contour linecolor, default 'grey'
    :param zorder_contour: zorder of the contour, default 1.

    :return: None
    """
    pie = np.pi if (pie is None) else pie

    if angle_fold_change <= 0 or angle_fold_change > 1:
        angle_fold_change = 1
        print('Warning: angle_fold_change should in (0, 1], otherwise will be set as 1.')
    x_angle, k = count_x(x_data_, angle_fold_change, angle_start_change, pie=pie)

    alpha = [1] * len(x_angle) if (alpha is None) else [1] + list(alpha)
    exposure = [0] * len(x_angle) if (exposure is None) else [0] + list(exposure)
    if color_list is None:
        color_list = [''] + ['lightpink', 'lightskyblue', 'wheat', 'mediumpurple'] * int(len(x_angle) // 4 + 1)
    else:
        color_list = [''] + list(color_list)

    label = [''] + list(label) if (label is not None) else False

    threshold_r_ = math.acos(r2 / r1)

    for i in range(1, len(x_data_) + 1):
        Control_center(x_angle, k, i, pie=pie, color_list=color_list, exposure=exposure, threshold_r=threshold_r_,
                       epsilon=epsilon, r1=r1, r2=r2, alpha=alpha, x0=x0, y0=y0, zorder=zorder_ring)
        if edge_open:
            edge(x_angle, k, i, pie=pie, x0=x0, y0=y0, lw=lw, lc=lc, r1=r1, r2=r2, zorder=zorder_edge)

        if label:
            if x_angle[i] - x_angle[i - 1] > number_threshold * pie:
                plot_label(x_angle, label, i, x_data_, pie=pie, exposure=exposure, x0=x0, y0=y0, fs=fs, fc=fc,
                           rotation=rotation, threshold=threshold, r1=r1, r2=r2, show_value=show_value,
                           show_percentage=show_percentage, fs2=fs2, fc2=fc2, zorder=zorder_label)
            else:
                plot_label(x_angle, label, i, x_data_, pie=pie, exposure=exposure, x0=x0, y0=y0, fs=fs, fc=fc,
                           rotation=rotation, threshold=threshold, r1=r1, r2=r2, show_value=False,
                           show_percentage=show_percentage, fs2=fs2, fc2=fc2, zorder=zorder_label)
    if r_list is not None:
        for r_ in r_list:
            contour(r_, figsize=figsize, xrange=xrange, linestyle=contour_ls, color=contour_lc, zorder=zorder_contour)


# for detailed plot
def Control_center_auto(x_data_, i, pie=None, color_list=None, exposure=None,
                        angle_fold_change=1, angle_start_change=0,
                        epsilon=0.001, r1=1, r2=0.8, alpha=None, x0=0, y0=0, zorder=10):

    pie = np.pi if (pie is None) else pie

    x_angle, k = count_x(x_data_, angle_fold_change=angle_fold_change, angle_start_change=angle_start_change, pie=pie)

    alpha = [1] * len(x_angle) if (alpha is None) else [1] + list(alpha)
    exposure = [0] * len(x_angle) if (exposure is None) else [0] + list(exposure)
    if color_list is None:
        color_list = [''] + ['lightpink', 'lightskyblue', 'wheat', 'mediumpurple'] * int(len(x_angle) // 4 + 1)
    else:
        color_list = [''] + list(color_list)

    # 1-#
    if x_angle[i - 1] < pie * 0.5:
        if x_angle[i] <= pie * 0.5:
            pi_1_1(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie:
            pi_1_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie + threshold_r:
            pi_1_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie * 1.5:
            pi_1_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            pi_1_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 2-#
    elif x_angle[i - 1] < pie:
        if x_angle[i] <= pie:
            pi_2_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie + threshold_r:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie * 1.5:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 3-#
    elif x_angle[i - 1] < pie * 1.5:
        if x_angle[i] <= pie * 1.5:
            pi_3_3(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            pi_3_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 4-#
    else:
        pi_4_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)


# 计算所需参数
def count_x(x_data_, angle_fold_change, angle_start_change, pie):
    x_data_ = np.array([0] + list(x_data_))
    x_alpha = angle_fold_change * x_data_ * 2 * pie / np.sum(x_data_)
    x_angle = [sum(x_alpha[:(i + 1)]) + angle_start_change for i in range(len(x_alpha))]
    k = [math.tan(i) for i in x_angle]
    x_angle[-1] = angle_fold_change * 2 * pie + angle_start_change

    return x_angle, k


# 绘制pie图边界线
def edge(x_angle, k, i, pie, x0=0, y0=0, lw=2, lc='white', r1=1, r2=0.8, zorder=11):
    if x_angle[i] == 0.5 * pie:
        plt.plot([x0, x0], [y0 + r2, y0 + r1], lw=lw, color=lc, zorder=11)

    elif x_angle[i] == 1.5 * pie:
        plt.plot([x0, x0], [y0 - r1, y0 - r2], lw=lw, color=lc, zorder=11)

    else:
        I1 = math.cos(x_angle[i]) / abs(math.cos(x_angle[i]))
        I2 = math.sin(x_angle[i]) / abs(math.sin(x_angle[i]))
        x1 = I1 * r1 / ((k[i] ** 2) + 1) ** 0.5
        y1 = I2 * (r1 ** 2 - x1 ** 2) ** 0.5
        x2 = I1 * r2 / ((k[i] ** 2) + 1) ** 0.5
        y2 = I2 * (r2 ** 2 - x2 ** 2) ** 0.5

        plt.plot([x0 + x2, x0 + x1], [y0 + y2, y0 + y1], lw=lw, color=lc, zorder=zorder)


# 绘制label
def plot_label(x_angle, label, i, x_data_, pie, exposure, x0, y0, fs, fc, rotation,
               threshold, r1, r2, show_value=False, show_percentage=False, fs2=15, fc2='black', zorder=12):

    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    r2_ = (r1 + r2) / 2
    delta_x = exposure[i] * r2_ * math.cos(angle)
    delta_y = exposure[i] * r2_ * math.sin(angle)

    if show_value and show_percentage:
        percentage = np.round(np.array([x_data_[j] / sum(x_data_) * 100 for j in range(len(x_data_))]), 3)

    if angle == 0.5 * pie:
        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = 90 * rotation
            if rotation:
                location = ['center', 'center']
            else:
                location = ['bottom', 'center']

        plt.text(x0 + delta_x, y0 + (1 + threshold) * r1 + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * delta_x, y0 + r2_ * (1 + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * delta_x, y0 + r2_ * (1 + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == pie:
        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = 0
            if rotation:
                location = ['center', 'center']
            else:
                location = ['center', 'right']
        plt.text(x0 - (1 + threshold) * r1 + delta_x, y0 + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle, verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 - r2_ * (r1 + delta_x), y0 + r2_ * delta_y, x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0, verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 - r2_ * (r1 + delta_x), y0 + r2_ * delta_y, str(percentage[i - 1]) + '%',
                     fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == 1.5 * pie:
        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = 270 * rotation
            if rotation:
                location = ['center', 'center']
            else:
                location = ['top', 'center']

        plt.text(x0 + delta_x, y0 - (1 + threshold) * r1 + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * delta_x, y0 - r2_ * (1 + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * delta_x, y0 - r2_ * (1 + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle < 0.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = rotation * angle * 180 / pie
            if rotation:
                location = ['center', 'center']
            else:
                location = ['baseline', 'left']
        x = (1 + threshold) * math.cos(angle) * r1
        y = (1 + threshold) * math.sin(angle) * r1
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)

    elif angle > 1.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y - delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y - delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = rotation * angle * 180 / pie
            if rotation:
                location = ['center', 'center']
            else:
                location = ['top', 'left']
        x = (1 + threshold) * math.cos(angle) * r1
        y = (1 + threshold) * math.sin(angle) * r1
        plt.text(x0 + x + delta_x, y0 + y - delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)

    elif pie < angle < 1.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y - delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y - delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = rotation * (180 + angle * 180 / pie)
            if rotation:
                location = ['center', 'center']
            else:
                location = ['top', 'right']
        x = (1 + threshold) * math.cos(angle) * r1
        y = (1 + threshold) * math.sin(angle) * r1
        plt.text(x0 + x - delta_x, y0 + y - delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)

    else:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        if rotation == 'center':
            rotation_angle = angle * 180 / pie - 90
            location = ['center', 'center']
        else:
            rotation_angle = rotation * (180 + angle * 180 / pie)
            if rotation:
                location = ['center', 'center']
            else:
                location = ['baseline', 'right']
        x = (1 + threshold) * math.cos(angle) * r1
        y = (1 + threshold) * math.sin(angle) * r1
        plt.text(x0 + x - delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation_angle,
                 verticalalignment=location[0], horizontalalignment=location[1], zorder=zorder)


# Draw contour lines
def contour(r1, figsize=None, xrange=None, linestyle='auto', color='grey', zorder=1):
    if figsize is None and linestyle == 'auto':
        figsize = 6
        print('Warning: missing the parameter \'figsize\' for auto linestyle!')
    if xrange is None and linestyle == 'auto':
        xrange = r1
        print('Warning: missing the parameter \'xrange\' for auto linestyle!')

    if linestyle == 'auto':  # 自动计算线长，使得虚线间隔合理
        coef = figsize * r1 / xrange / 10
        linestyle = (98 * coef, (191 * coef, 20 * coef))

    epsilon = 0.001
    x1 = np.arange(-r1, r1, epsilon)
    x2 = np.arange(-r1, r1, epsilon)
    y1 = (r1 ** 2 - x1 ** 2) ** 0.5
    y2 = -(r1 ** 2 - x2 ** 2) ** 0.5
    plt.plot(x1, y1, linestyle=linestyle, color=color, zorder=zorder)
    plt.plot(x2, y2, linestyle=linestyle, color=color, zorder=zorder)


# ring 主体
def pi_1_1(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r2 * math.cos(x_angle[i])
    x2 = r2 * math.cos(x_angle[i - 1])
    x3 = r1 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i - 1])

    ab = 1
    if x2 > x3:
        x2, x3 = x3, x2
        ab = 0

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    if ab:
        y_plot_f_u = k[i] * x_plot_f
        y_plot_f_d = (r2 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_m_u = k[i] * x_plot_m
        y_plot_m_d = k[i - 1] * x_plot_m
        y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
        y_plot_b_d = k[i - 1] * x_plot_b
    else:
        y_plot_f_u = k[i] * x_plot_f
        y_plot_f_d = (r2 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
        y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r1 * math.cos(x_angle[i])
    x2 = r2 * math.cos(x_angle[i])
    x3 = r2 * math.cos(x_angle[i - 1])
    x4 = r1 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = k[i] * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i - 1])
    x4 = r1 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = 0 * x_plot_m
    y_plot_m_d = k[i] * x_plot_m
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i - 1])
    x4 = r1 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i])
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_1_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i - 1])
    x4 = r1 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = (r1 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i - 1] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                         color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r1 * math.cos(x_angle[i])
    x2 = r2 * math.cos(x_angle[i])
    x3 = r1 * math.cos(x_angle[i - 1])
    x4 = r2 * math.cos(x_angle[i - 1])

    ab = 1
    if x2 > x3:
        x2, x3 = x3, x2
        ab = 0

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    if ab:
        y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_f_d = k[i] * x_plot_f
        y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_b_u = k[i - 1] * x_plot_b
        y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5
    else:
        y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_f_d = k[i] * x_plot_f
        y_plot_m_u = k[i - 1] * x_plot_m
        y_plot_m_d = k[i] * x_plot_m
        y_plot_b_u = k[i - 1] * x_plot_b
        y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_a_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i - 1])
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = 0 * x_plot_m
    y_plot_m_d = k[i] * x_plot_m
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_b_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i - 1])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = k[i - 1] * x_plot_m
    y_plot_m_d = 0 * x_plot_m
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = 0 * x_plot_m
    y_plot_m_d = k[i] * x_plot_m
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_a_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i - 1])
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i])
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_b_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i - 1])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = k[i - 1] * x_plot_m
    y_plot_m_d = 0 * x_plot_m
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i])
    x4 = r2 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
    y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_a_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = - r2
    x3 = r1 * math.cos(x_angle[i - 1])
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_2_b_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = - r1
    x2 = r1 * math.cos(x_angle[i - 1])
    x3 = - r2
    x4 = r2 * math.cos(x_angle[i - 1])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_f_d = 0 * x_plot_f
    y_plot_m_u = k[i - 1] * x_plot_m
    y_plot_m_d = 0 * x_plot_m
    y_plot_b_u = k[i - 1] * x_plot_b
    y_plot_b_d = (r2 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)

    x1 = - r1
    x2 = - r2
    x3 = r2 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = 0 * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_3_3(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r1 * math.cos(x_angle[i - 1])
    x2 = r1 * math.cos(x_angle[i])
    x3 = r2 * math.cos(x_angle[i - 1])
    x4 = r2 * math.cos(x_angle[i])

    ab = 1
    if x2 > x3:
        x2, x3 = x3, x2
        ab = 0

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    if ab:
        y_plot_f_u = k[i - 1] * x_plot_f
        y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_m_u = k[i - 1] * x_plot_m
        y_plot_m_d = k[i] * x_plot_m
        y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
        y_plot_b_d = k[i] * x_plot_b
    else:
        y_plot_f_u = k[i - 1] * x_plot_f
        y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_b_u = - (r2 ** 2 - x_plot_b ** 2) ** 0.5
        y_plot_b_d = k[i] * x_plot_b

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_3_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r1 * math.cos(x_angle[i - 1])
    x2 = r2 * math.cos(x_angle[i - 1])
    x3 = r2 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i])

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    y_plot_f_u = k[i - 1] * x_plot_f
    y_plot_f_d = - (r1 ** 2 - x_plot_f ** 2) ** 0.5
    y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
    y_plot_b_u = k[i] * x_plot_b
    y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pi_4_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder):
    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    delta_x = exposure[i] * r1 * math.cos(angle)
    delta_y = exposure[i] * r1 * math.sin(angle)

    x1 = r2 * math.cos(x_angle[i - 1])
    x2 = r1 * math.cos(x_angle[i - 1])
    x3 = r2 * math.cos(x_angle[i])
    x4 = r1 * math.cos(x_angle[i])

    ab = 1
    if x2 > x3:
        x2, x3 = x3, x2
        ab = 0

    x_plot_f = np.arange(x1, x2, epsilon)
    x_plot_m = np.arange(x2, x3, epsilon)
    x_plot_b = np.arange(x3, x4, epsilon)

    if ab:
        y_plot_f_u = - (r2 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_f_d = k[i - 1] * x_plot_f
        y_plot_m_u = - (r2 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_m_d = - (r1 ** 2 - x_plot_m ** 2) ** 0.5
        y_plot_b_u = k[i] * x_plot_b
        y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5
    else:
        y_plot_f_u = - (r2 ** 2 - x_plot_f ** 2) ** 0.5
        y_plot_f_d = k[i - 1] * x_plot_f
        y_plot_m_u = k[i] * x_plot_m
        y_plot_m_d = k[i - 1] * x_plot_m
        y_plot_b_u = k[i] * x_plot_b
        y_plot_b_d = - (r1 ** 2 - x_plot_b ** 2) ** 0.5

    x_plot = np.append(x_plot_f, x_plot_m)
    y_plot_u = np.append(y_plot_f_u, y_plot_m_u)
    y_plot_d = np.append(y_plot_f_d, y_plot_m_d)

    x_plot = np.append(x_plot, x_plot_b)
    y_plot_u = np.append(y_plot_u, y_plot_b_u)
    y_plot_d = np.append(y_plot_d, y_plot_b_d)

    plt.fill_between(x_plot + x0 + delta_x, y_plot_u + y0 + delta_y, y_plot_d + y0 + delta_y,
                     color=color_list[i], alpha=alpha[i], zorder=zorder)


def pie_plot(epsilon=0.001, color='gray', r1=1, alpha=1, x0=0, y0=0, zorder=10):
    x_plot = np.arange(-r1, r1, epsilon)
    y_plot_u = (1 - x_plot ** 2) ** 0.5
    y_plot_d = - (1 - x_plot ** 2) ** 0.5

    plt.fill_between(x_plot + x0, y_plot_u + y0, y_plot_d + y0,
                     color=color, alpha=alpha, zorder=zorder)


def Control_center(x_angle, k, i, threshold_r, pie, color_list=False, exposure=False,
                   epsilon=0.001, r1=1, r2=0.8, alpha=False, x0=0, y0=0, zorder=10):

    # 0-#
    if (len(x_angle) == 2) and (r2 == 0):
        pie_plot(epsilon=epsilon, color=color_list[1], r1=r1, alpha=alpha[1], x0=x0, y0=y0, zorder=zorder)

    # 1-#
    elif x_angle[i - 1] < pie * 0.5:
        if x_angle[i] <= pie * 0.5:
            pi_1_1(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie:
            pi_1_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie + threshold_r:
            pi_1_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie * 1.5:
            pi_1_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            pi_1_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 2-#
    elif x_angle[i - 1] < pie:
        if x_angle[i] <= pie:
            pi_2_2(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie + threshold_r:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_3_a(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        elif x_angle[i] <= pie * 1.5:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_3_b(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            if x_angle[i - 1] < pie - threshold_r:
                pi_2_a_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
            else:
                pi_2_b_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 3-#
    elif x_angle[i - 1] < pie * 1.5:
        if x_angle[i] <= pie * 1.5:
            pi_3_3(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)
        else:
            pi_3_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)

    # 4-#
    else:
        pi_4_4(x_angle, k, i, exposure, epsilon, color_list, r1, r2, alpha, x0, y0, zorder)


# 绘制pie图边界线
def edge_auto(x_data_, i, pie=None,
              x0=0, y0=0, lw=2, lc='white', r1=1, r2=0.8,
              angle_fold_change=1, angle_start_change=0, zorder=11):

    pie = np.pi if (pie is None) else pie

    x_angle, k = count_x(x_data_, angle_fold_change=angle_fold_change, angle_start_change=angle_start_change, pie=pie)

    if x_angle[i] == 0.5 * pie:
        plt.plot([x0, x0], [y0, y0+r], lw=lw, color=lc, zorder=11)
    elif x_angle[i] == 1.5 * pie:
        plt.plot([x0, x0], [y0-r, y0], lw=lw, color=lc, zorder=11)
    else:
        I1 = math.cos(x_angle[i]) / abs(math.cos(x_angle[i]))
        I2 = math.sin(x_angle[i]) / abs(math.sin(x_angle[i]))
        x1 = I1 * r1 / ((k[i] ** 2) + 1) ** 0.5
        y1 = I2 * (r1 ** 2 - x1 ** 2) ** 0.5
        x2 = I1 * r2 / ((k[i] ** 2) + 1) ** 0.5
        y2 = I2 * (r2 ** 2 - x2 ** 2) ** 0.5

        plt.plot([x0 + x2, x0 + x1], [y0 + y2, y0 + y1], lw=lw, color=lc, zorder=zorder)


# 绘制label
def plot_label_auto(x_data_, label, i, pie=None, exposure=None, x0=0, y0=0, fs=15, fc='black', rotation=True,
                    threshold=0.05, r1=1, r2=0.8, show_value=False, show_percentage=False, fs2=15, fc2='black',
                    angle_fold_change=1, angle_start_change=0, zorder=12):

    pie = np.pi if (pie is None) else pie

    x_angle, k = count_x(x_data_, angle_fold_change, angle_start_change, pie=pie)

    exposure = [0] * len(x_angle) if (exposure is None) else [0] + list(exposure)
    label = [''] + list(label)

    angle = 0.5 * (x_angle[i - 1] + x_angle[i])
    r2_ = (r1 + r2) / 2
    delta_x = exposure[i] * r2_ * math.cos(angle)
    delta_y = exposure[i] * r2_ * math.sin(angle)

    if show_value and show_percentage:
        percentage = np.round(np.array([x_data_[j] / sum(x_data_) * 100 for j in range(len(x_data_))]), 3)

    if angle == 0.5 * pie:
        plt.text(x0 + delta_x, y0 + (1 + threshold) * r1 + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=90 * rotation,
                 verticalalignment="bottom", horizontalalignment="center", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * delta_x, y0 + r2_ * (r1 + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * delta_x, y0 + r2_ * (r1 + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == pie:
        plt.text(x0 - (1 + threshold) * r1 + delta_x, y0 + delta_y, label[i], fontsize=fs, color=fc, rotation=0,
                 verticalalignment="center", horizontalalignment="right", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 - r2_ * (r1 + delta_x), y0 + r2_ * delta_y, x_data_[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 - r2_ * (r1 + delta_x), y0 + r2_ * delta_y, str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle == 1.5 * pie:
        plt.text(x0 + delta_x, y0 - (1 + threshold) * r1 + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=270 * rotation,
                 verticalalignment="top", horizontalalignment="center", zorder=zorder)
        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * delta_x, y0 - r2_ * (r1 + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * delta_x, y0 - r2_ * (r1 + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

    elif angle < 0.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r1
        y += threshold * math.sin(angle) * r1
        plt.text(x0 + x + delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * angle * 180 / pie,
                 verticalalignment="baseline", horizontalalignment="left", zorder=zorder)

    elif angle > 1.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y - delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x + delta_x), y0 + r2_ * (y - delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r1
        y += threshold * math.sin(angle) * r1
        plt.text(x0 + x + delta_x, y0 + y - delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * angle * 180 / pie,
                 verticalalignment="top", horizontalalignment="left", zorder=zorder)

    elif pie < angle < 1.5 * pie:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y - delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y - delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r1
        y += threshold * math.sin(angle) * r1
        plt.text(x0 + x - delta_x, y0 + y - delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * (180 + angle * 180 / pie),
                 verticalalignment="top", horizontalalignment="right", zorder=zorder)

    else:
        x = math.cos(angle)
        y = math.sin(angle)

        if show_value and (not show_percentage):
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y + delta_y), x_data_[i - 1], fontsize=fs2, color=fc2,
                     rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)
        elif show_value and show_percentage:
            plt.text(x0 + r2_ * (x - delta_x), y0 + r2_ * (y + delta_y), str(percentage[i - 1]) + '%', fontsize=fs2,
                     color=fc2, rotation=0,
                     verticalalignment="center", horizontalalignment="center", zorder=zorder)

        x += threshold * math.cos(angle) * r1
        y += threshold * math.sin(angle) * r1
        plt.text(x0 + x - delta_x, y0 + y + delta_y, label[i], fontsize=fs, color=fc,
                 rotation=rotation * (180 + angle * 180 / pie),
                 verticalalignment="baseline", horizontalalignment="right", zorder=zorder)


# 贝塞尔曲线部分
def Bezier_function(P0_, P1_, P2_, t_):
    return (1 - t_) ** 2 * P0_ + 2 * t_ * (1 - t_) * P1_ + t_ ** 2 * P2_


# 生成贝塞尔曲线的点集
def Bezier_data(P0_, P1_, P2_, dpi=1000):
    return np.array([Bezier_function(P0_, P1_, P2_, i / dpi) for i in range(dpi + 1)])


# 根据角度绘图，专用于circos plot
def Bezier_curve(angel1_, angel2_, r_, x0=0, y0=0, dpi=1000, curvature=1, **kwargs):
    x1_ = r_ * math.cos(angel1_) + x0
    y1_ = r_ * math.sin(angel1_) + y0
    x2_ = r_ * math.cos(angel2_) + x0
    y2_ = r_ * math.sin(angel2_) + y0

    if curvature != 0:
        P0_ = np.array([x1_, y1_])
        P2_ = np.array([x2_, y2_])

        M0_ = (P0_ + P2_) / 2
        C0_ = np.array([x0, y0])
        delta_ = C0_ - M0_
        P1_ = C0_ - (1 - curvature) * delta_
        curve_data_ = Bezier_data(P0_, P1_, P2_, dpi=dpi)

        plt.plot(curve_data_[:, 0], curve_data_[:, 1], **kwargs)
    else:
        plt.plot([x1_, x2_], [y1_, y2_], **kwargs)


# handsome
def handsome_Bezier_curve(angel1_, angel2_, r_, x0=0, y0=0, dpi=1000, curvature=1, **kwargs):
    x1_ = r_ * math.cos(angel1_) + x0
    y1_ = r_ * math.sin(angel1_) + y0
    x2_ = r_ * math.cos(angel2_) + x0
    y2_ = r_ * math.sin(angel2_) + y0

    if curvature != 0:
        P0_ = np.array([x1_, y1_])
        P2_ = np.array([x2_, y2_])

        M0_ = (P0_ + P2_) / 2
        C0_ = np.array([x0, y0])
        delta_ = C0_ - M0_
        P1_ = C0_ - (1 - curvature) * delta_
        curve_data_ = Bezier_data(P0_, P1_, P2_, dpi=dpi)

        plt.plot(curve_data_[:, 0], curve_data_[:, 1], **kwargs)
    else:
        plt.plot([x1_, x2_], [y1_, y2_], **kwargs)


def ring_Bezier_curve(x_data_, position_pair, r_, x0=0, y0=0, dpi=1000, curvature=1,
                      angle_fold_change=1, angle_start_change=0, pie=False, **kwargs):
    if not pie:
        pie = np.pi
    x_angle_, k_ = count_x(x_data_, angle_fold_change, angle_start_change, pie=pie)

    for pair_ in position_pair:
        angle_1_ = (x_angle_[pair_[0]] + x_angle_[pair_[0] + 1]) / 2
        angle_2_ = (x_angle_[pair_[1]] + x_angle_[pair_[1] + 1]) / 2

        Bezier_curve(angle_1_, angle_2_, r_, x0=x0, y0=y0, dpi=dpi, curvature=curvature, **kwargs)


def ring_line_plot(x_data_, r1=0.8, max_height=0.15, x0=0, y0=0,
                      angle_fold_change=1, angle_start_change=0, pie=False, **kwargs):
    if not pie:
        pie = np.pi
    _length = len(x_data_)
    x_angle_, k_ = count_x([1] * _length, angle_fold_change, angle_start_change, pie=pie)

    norm_x_data_ = x_data_ / np.max(abs(x_data_))
    for h_, angle_ in zip(norm_x_data_, x_angle_):
        add_r = h_ * max_height
        x1_ = r1 * math.cos(angle_) + x0
        y1_ = r1 * math.sin(angle_) + y0
        x2_ = (r1 + add_r) * math.cos(angle_) + x0
        y2_ = (r1 + add_r) * math.sin(angle_) + y0

        plt.plot([x1_, x2_], [y1_, y2_], **kwargs)


def interval(list_like_, object_):
    out_ = []
    for i_ in list_like_:
        out_ += [object_, i_]

    return out_


def take_angle(start, end, num=100):
    return [start + i_ * (end - start) / (num + 2) for i_ in range(1, num + 1)]


# demo
if __name__ == "__main__":
    x_data = [1]
    ring_plot(x_data, r2=0)
    plt.show()
