#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib import colors
from collections import defaultdict
import math


def coordinate_factory():
    return [0, 0]


def linkage_factory():
    return [0, 0, '']


def mirror(array_, start_, end_):
    return start_ + end_ - array_


def sort_set_list(list_data):
    set_list = list(set(list_data))
    set_list.sort(key=list_data.index)

    return set_list


def plt_sub_tree(plt_, nodes_, dict_, c, reverse, y0_, y1_, **kwargs):
    left_ = dict_[nodes_[0]]
    right_ = dict_[nodes_[1]]
    height_ = dict_[nodes_[2]][1]
    x_ = np.array([left_[0], left_[0], right_[0], right_[0]])
    y_ = np.array([left_[1], height_, height_, right_[1]])

    if reverse == 'left':
        x_, y_ = mirror(y_, y0_, y1_), x_
    elif reverse == 'right':
        x_, y_ = y_, x_
    elif reverse == 'bottom':
        x_, y_ = x_, mirror(y_, y0_, y1_)

    plt_.plot(x_, y_, c=c, **kwargs)


def find_son(Tree_, node_):
    n_col = np.argmax(np.sum(Tree_ == node_))
    n_row = Tree_.index[Tree_.iloc[:, n_col] == node_]

    return Tree_.iloc[n_row, (n_col + 1):]


def hierarchical_plot(plt_, data_, metric='euclidean', method='ward', info_only=False,
                      cluster_num=3, height_threshold=None, reverse=False, color_list=False, label=False,
                      label_rotation=0, label_fontsize=12, x0=0, x1=10, y0=0, y1=6, bgcolor='black',
                      return_coordinate=False, **kwargs):

    if plt_ == plt:
        plt_ = plt_.gca()

    if not color_list:
        color = list(colors.CSS4_COLORS)
        pick_col = [9, 11, 25, 40, 44, 83, 96, 105, 106]
        color_list = [color[i] for i in pick_col]
    if reverse == 'top':
        reverse = False

    disMat = sch.distance.pdist(X=data_, metric=metric)
    Z = sch.linkage(disMat, method=method)

    # 定义字典
    coordinate_dict = defaultdict(coordinate_factory)
    linkage_dict = defaultdict(linkage_factory)
    reverse_dict = defaultdict(int)

    # 将聚类后的新index作为index
    linkage = pd.DataFrame(Z)
    linkage.index = list(range(len(Z) + 1, len(Z) * 2 + 1))

    # 确定类别数与分界点
    if not cluster_num:
        cluster_num = linkage.loc[linkage.iloc[:, 2] >= height_threshold].shape[0]
    split = linkage.iloc[-(cluster_num - 1):]

    top_nodes = set(split.index)
    bottom_nodes = set([split.iloc[i, j] for i in range(split.shape[0]) for j in range(2)])
    bottom_nodes = bottom_nodes - top_nodes

    # 改动横纵坐标准备bottom
    delta_x = x1 - x0
    delta_y = y1 - y0

    if reverse not in ['left', 'right', 'bottom', False]:
        raise ValueError('reverse option should be set as \'left\', \'right\', \'bottom\' or False')

    if reverse in ['left', 'right']:
        delta_x, delta_y = delta_y, delta_x
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    stride = delta_x / (len(Z) + 1)
    multi_y = delta_y / np.max(linkage.iloc[:, 2])

    # 提取纵坐标信息
    for ind in linkage.index:
        value = linkage.loc[ind]
        linkage_dict[ind] = [int(value[0]), int(value[1]), '']
        reverse_dict[int(value[0])] = ind
        reverse_dict[int(value[1])] = ind

        if int(value[0]) not in coordinate_dict.keys():
            coordinate_dict[int(value[0])] = [0, y0]
        if int(value[1]) not in coordinate_dict.keys():
            coordinate_dict[int(value[1])] = [0, y0]
        coordinate_dict[ind] = [0, value[2] * multi_y + y0]

    reverse_dict[ind] = ind

    # 确认层级结构
    level = [list(range(len(Z) + 1))]
    level_info = list(range(len(Z) + 1))
    while len(set(level_info)) > 1:
        level_info = [reverse_dict[i] for i in level_info]
        level += [level_info]
    tree_level = pd.DataFrame(level).T.sort_values(by=list(range(len(level)))[::-1])
    tree_level = tree_level.reset_index(drop=True)

    Tree = []
    for value in tree_level.values:
        Tree += [sort_set_list(list(value))[:: -1]]
    Tree = pd.DataFrame(Tree).sort_values(by=list(range(len(level))))
    Tree = Tree.reset_index(drop=True)

    # 定义横坐标
    for node in range(len(Z) + 1):
        x_ = np.mean(list(Tree.index[np.sum(Tree == node, axis=1) != 0]))  # 寻找node所在行
        coordinate_dict[node][0] = x_
    for node in linkage_dict.keys():
        sons = linkage_dict[node]
        x_ = (coordinate_dict[sons[0]][0] + coordinate_dict[sons[1]][0]) / 2
        coordinate_dict[node][0] = x_
    for node in coordinate_dict.keys():
        coordinate_dict[node][0] = coordinate_dict[node][0] * stride + x0

    # 给定颜色
    for top_node in top_nodes:
        if top_node not in linkage_dict.keys():
            continue
        linkage_dict[top_node][2] = bgcolor

    for rank_, bottom_node in enumerate(bottom_nodes):
        if bottom_node not in linkage_dict.keys():
            continue
        sub_tree = find_son(Tree, bottom_node)
        cluster_nodes = set([node for value in sub_tree.values for node in value if node == node]) | {bottom_node}
        for cluster_node in cluster_nodes:
            if cluster_node not in linkage_dict.keys():
                continue
            linkage_dict[cluster_node][2] = color_list[rank_]

    if info_only:
        return coordinate_dict, linkage_dict

    # 画图
    for key in linkage.index[::-1]:
        nodes = linkage_dict[key][:2] + [key]
        c = linkage_dict[key][2]
        plt_sub_tree(plt_, nodes, coordinate_dict, c, reverse, y0, y1, **kwargs)

    if label is not None:
        if type(label) == bool:
            label = list(range(len(Z) + 1))
        if not reverse:
            plt_.set_xticks([coordinate_dict[i][0] for i in range(len(Z) + 1)], minor=label)
            plt_.xaxis.set_tick_params(rotation=label_rotation, size=label_fontsize)
        elif reverse in ['left', 'right']:
            if reverse == 'left':
                plt_.yaxis.set_ticks_position('right')
            plt_.set_yticks([coordinate_dict[i][0] for i in range(len(Z) + 1)], minor=label)
            plt_.yaxis.set_tick_params(rotation=label_rotation, size=label_fontsize)
        else:
            plt_.set_xticks([coordinate_dict[i][0] for i in range(len(Z) + 1)], minor=label)
            plt_.xaxis.set_ticks_position('top')
            plt_.xaxis.set_tick_params(rotation=label_rotation, size=label_fontsize)

    if return_coordinate:
        return [[i, coordinate_dict[i][0]] for i in range(len(Z) + 1)]


def trans_2_alpha_r(dict_, real_r_, angle_fold_change, angle_start_change, ter_r0=0.1):
    coordinate_circos_ = {}
    for key, value in zip(dict_.keys(), dict_.values()):
        coordinate_circos_[key] = [angle_fold_change * value[0] * 2 * pi + angle_start_change,
                                   (1 - value[1]) * real_r_ + ter_r0]

    return coordinate_circos_


def plt_sub_circos_tree(plt_, nodes_, dict_, c, x0_, y0_, **kwargs):
    alpha0_ = dict_[nodes_[0]][0]
    alpha1_ = dict_[nodes_[1]][0]
    r0_ = dict_[nodes_[0]][1]
    r1_ = dict_[nodes_[1]][1]
    r2_ = dict_[nodes_[2]][1]

    alphas = np.arange(alpha0_, alpha1_, 0.001)
    x_ = np.array([math.cos(angle) * r2_ for angle in alphas]) + x0_
    y_ = np.array([math.sin(angle) * r2_ for angle in alphas]) + y0_

    xx0_ = np.array([math.cos(alpha0_) * r2_, math.cos(alpha0_) * r0_]) + x0_
    yy0_ = np.array([math.sin(alpha0_) * r2_, math.sin(alpha0_) * r0_]) + y0_

    xx1_ = np.array([math.cos(alpha1_) * r2_, math.cos(alpha1_) * r1_]) + x0_
    yy1_ = np.array([math.sin(alpha1_) * r2_, math.sin(alpha1_) * r1_]) + y0_

    line_dict = {}
    if 'marker' in kwargs.keys():
        scatter_dict = {}
        for key in kwargs.keys():
            if 'marker' in key:
                if 'markersize' in key:
                    scatter_dict['s'] = kwargs[key]
                else:
                    scatter_dict[key] = kwargs[key]
            else:
                line_dict[key] = kwargs[key]
        for i_ in range(2):
            plt_.scatter(xx0_[i_], yy0_[i_], c=c, **scatter_dict)
            plt_.scatter(xx1_[i_], yy1_[i_], c=c, **scatter_dict)

    plt_.plot(x_, y_, c=c, **line_dict)
    plt_.plot(xx0_, yy0_, c=c, **line_dict)
    plt_.plot(xx1_, yy1_, c=c, **line_dict)

    return line_dict


def circos_hierarchical_clustering(plt_, data_, cluster_r0=0.1, cluster_r=1, x0=0, y0=0,
                                   metric='euclidean', method='ward', cluster_num=3, color_list=False, bgcolor='black',
                                   angle_fold_change=1, angle_start_change=0, **kwargs):
    real_cluster_r = cluster_r - cluster_r0
    coordinate, linkage = hierarchical_plot(plt_, data_, metric=metric, method=method, cluster_num=cluster_num,
                                            info_only=True, height_threshold=None, color_list=color_list,
                                            x0=0, x1=1, y0=0, y1=1, bgcolor=bgcolor, return_coordinate=False)

    coordinate_circos = trans_2_alpha_r(coordinate, real_cluster_r,
                                        angle_fold_change, angle_start_change, ter_r0=cluster_r0)

    for key in list(linkage.keys()):
        nodes = linkage[key][:2] + [key]
        c = linkage[key][2]
        line_dict = plt_sub_circos_tree(plt_, nodes, coordinate_circos, c, x0, y0, **kwargs)

    origin = list(linkage.keys())[-1]
    alpha_origin = coordinate_circos[origin][0]
    r_origin = coordinate_circos[origin][1]
    c = linkage[origin][2]
    x_origin = math.cos(alpha_origin) * r_origin + x0
    y_origin = math.sin(alpha_origin) * r_origin + y0
    plt_.plot([x0, x_origin], [y0, y_origin], c=c, **line_dict)

    return plt_


pi = np.pi
