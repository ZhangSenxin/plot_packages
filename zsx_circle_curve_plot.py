# -*- coding: UTF-8 -*-
# @Author: Zhang Senxin

import numpy as np
import math


def mod_fun(array_):
    return np.dot(array_, array_) ** 0.5


def circle_curve_plot(plt_, x1, x2, y1=0, y2=0, up_down='up', alpha=180, epsilon=1000, **kwargs):
    """
    need math
    """
    up_down = 1 if up_down in 'up' else -1

    alpha0 = alpha / 180 * math.pi
    l = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    h = (l / 2) * (1 / math.tan(alpha0 / 2))

    beta = math.atan((y2 - y1) / (x2 - x1)) if x2 != x1 else math.pi / 2

    gamma = abs(math.pi / 2 - beta)

    if beta == 0:
        dieta_x = 0
    elif beta <= math.pi / 2:
        dieta_x = h * math.cos(gamma)
    elif beta > math.pi / 2:
        dieta_x = - h * math.cos(gamma)

    dieta_y = - (h * math.sin(gamma)) if (beta != math.pi / 2) else 0

    x0 = (x1 + x2) / 2 + up_down * dieta_x
    y0 = (y1 + y2) / 2 + up_down * dieta_y

    r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    array1 = np.array([x1 - x0, y1 - y0])
    array2 = np.array([x2 - x0, y2 - y0])
    x_axis = np.array([1, 0])

    alpha1 = math.acos(np.dot(array1, x_axis) / (mod_fun(array1) * mod_fun(x_axis)))
    alpha2 = math.acos(np.dot(array2, x_axis) / (mod_fun(array2) * mod_fun(x_axis)))

    alpha1 = alpha1 if array1[1] >= 0 else 2 * math.pi - alpha1
    alpha2 = alpha2 if array2[1] >= 0 else 2 * math.pi - alpha2

    alpha1, alpha2 = min([alpha1, alpha2]), max([alpha1, alpha2])

    # circle
    cis = True if up_down == 1 else False
    if alpha2 - alpha1 < math.pi:
        cis = True

    elif alpha2 - alpha1 > math.pi:
        cis = False

    elif beta == math.pi / 2:
        pass

    elif alpha1 > math.pi / 2:
        cis = not cis

    if cis:
        alpha_range1 = np.arange(alpha1, alpha2, math.pi / epsilon)
    else:
        alpha_range1_1 = np.arange(alpha2, 2 * math.pi, math.pi / epsilon)
        alpha_range1_2 = np.arange(math.pi / epsilon, alpha1, math.pi / epsilon)
        alpha_range1 = np.concatenate([alpha_range1_1, alpha_range1_2])

    x_list = [r * math.cos(angle) + x0 for angle in alpha_range1]
    y_list = [r * math.sin(angle) + y0 for angle in alpha_range1]

    plt_.plot(x_list, y_list, **kwargs)
