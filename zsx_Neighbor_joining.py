#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from collections import OrderedDict, defaultdict, namedtuple


def default_one_factory():
    return 1


def _genarate_spots(_n, _k):
    return np.random.multivariate_normal([0] * _k, np.diag([1] * _k), _n)


def _check_cmod(c, cmod):
    if c is not None and cmod not in [1, 2, 3]:
        raise ValueError('Error! Element in parameter \'cmod\' is Illegal! Please use 1,2 or 3 to fill it.')


def measure_color(c, cmod, c_alpha_min, n_sample):
    if c is None:
        c = ['Greys']
    if cmod is None:
        cmod = 2
    if c is not None and type(c) != list:
        c = [c]

    if not c_alpha_min:
        if cmod == 2:
            c_alpha_min = 0.5
        else:
            c_alpha_min = 1
    if c_alpha_min > 1:
        c_alpha_min = 1
    elif c_alpha_min < 0:
        c_alpha_min = 0

    if cmod == 1:
        color = c[0]
        if c_alpha_min == 1:
            colors_ = [color] * n_sample
        else:
            if type(color) == str:
                color = colors.to_rgba_array(color)[0]
            colors_ = []
            for i_ in range(n_sample):
                alpha = ((1 - c_alpha_min) * (n_sample - i_) / n_sample + c_alpha_min) * (1 - 1e-6)
                colour = color.copy()
                colour[3] = alpha
                colors_ += [colour]

    elif cmod == 2:
        try:
            cmap1 = cm.get_cmap(c[0])
        except ValueError:
            cmap1 = c[0]
        colors_ = [cmap1(((1 - c_alpha_min) * (n_sample - i_) / n_sample + c_alpha_min) * (1 - 1e-6))
                   for i_ in range(n_sample)]

    elif cmod == 3:
        c_len = len(c)
        color_touch = int(np.ceil(n_sample / c_len))
        colors_ = []
        for colour in c:
            colors_ += [colour] * color_touch

    return colors_


def compute_squared_EDM_method(X):
    m, n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))

    return (H + H.T - 2 * G) ** 0.5


def _matrix_Q_compute_old(matrix_):
    n_ = len(matrix_)
    matrix_Q_ = np.zeros([n_, n_])
    sum_mat_ = list(np.sum(matrix_, axis=1))
    for i_ in range(n_ - 1):
        for j_ in range(i_ + 1, n_):
            q_val_ = (len(matrix_) - 2) * matrix_.iloc[i_, j_] - sum_mat_[i_] - sum_mat_[j_]
            matrix_Q_[i_, j_] = q_val_
            matrix_Q_[j_, i_] = q_val_
            
    return matrix_Q_, sum_mat_


def matrix_Q_compute(matrix_):
    matrix_ = np.array(matrix_)
    n_ = len(matrix_)
    sum_mat_ = list(np.sum(matrix_, axis=1))
    sum_mat_mat = np.array([sum_mat_ for _ in range(len(sum_mat_))])

    matrix_Q_ = (len(matrix_) - 2) * matrix_ - sum_mat_mat - sum_mat_mat.T

    for i_ in range(n_):
        matrix_Q_[i_, i_] = 0

    return matrix_Q_, sum_mat_


def update_distance_df(distance_df_, node1_, node2_, dist_node12_, start_id_):
    columns_ = list(distance_df_.columns)
    columns_.remove(node1_)
    columns_.remove(node2_)

    if columns_:
        add_col = list((distance_df_.loc[node1_, columns_] + distance_df_.loc[node2_, columns_] -
                        dist_node12_) / 2) + [0]
        distance_df_small_ = distance_df_.loc[columns_, columns_]
        distance_df_small_.loc[:, start_id_] = add_col[: -1]
        distance_df_small_.loc[start_id_, :] = add_col
    else:
        distance_df_small_ = pd.DataFrame([0])

    return distance_df_small_


def compute_delta(reverse, x_lim, y_lim, x_max, y_max):
    delta_x = x_max - x_lim
    delta_y = y_max - y_lim
    if reverse in ['left', 'right']:
        delta_x, delta_y = delta_y, delta_x
        x_lim, y_lim = y_lim, x_lim
        x_max, y_max = y_max, x_max

    return delta_x, delta_y, x_lim, y_lim, x_max, y_max


def core_process(info_, exist_node_dict_, num_dict_, sort=True):
    x_start_, y_start_, strand_, _ = exist_node_dict_[info_[0]]

    node_info_ = info_[1][0]
    whole_num_ = info_[1][1]
    node1_, node2_ = list(node_info_.keys())
    num1_, num2_ = num_dict_[node1_], num_dict_[node2_]

    if sort:
        if (strand_ == 'right' and num1_ < num2_) or (strand_ == 'left' and num1_ > num2_):
            node1_, node2_ = node2_, node1_
            num1_, num2_ = num_dict_[node1_], num_dict_[node2_]
        elif num1_ == num2_:
            if (strand_ == 'right' and node_info_[node1_] < node_info_[node2_]) or \
                    (strand_ == 'left' and node_info_[node1_] > node_info_[node2_]):
                node1_, node2_ = node2_, node1_
                num1_, num2_ = num_dict_[node1_], num_dict_[node2_]

    x1_ = x_start_ - num2_ / 2
    x2_ = x_start_ + num1_ / 2
    y1_ = - node_info_[node1_] + y_start_
    y2_ = - node_info_[node2_] + y_start_

    return x1_, x2_, y1_, y2_, node1_, node2_, y_start_


def mirror(array_, start_, end_):
    return start_ + end_ - array_


def plt_sub_tree(plt_, x_, y_, c, reverse, y0_, y1_, show_node=False, color=False,
                 scatter_size=50, marker='D', **kwargs):
    if reverse == 'left':
        x_, y_ = mirror(y_, y0_, y1_), x_
    elif reverse == 'right':
        x_, y_ = y_, x_
    elif reverse == 'bottom':
        x_, y_ = x_, mirror(y_, y0_, y1_)

    plt_.plot(x_, y_, c=c, **kwargs)

    if show_node:
        plt_.scatter(x_[0], y_[0], c=color[0], s=scatter_size, marker=marker, zorder=10)
        plt_.scatter(x_[-1], y_[-1], c=color[1], s=scatter_size, marker=marker, zorder=10)


def text_pos_tune(reverse, outer):
    x_outer = 0
    y_outer = 0
    ver_pos = "top"
    hor_pos = "center"

    if reverse == 'left':
        x_outer = outer
        ver_pos = 'center'
        hor_pos = 'left'
    elif reverse == 'right':
        x_outer = -outer
        ver_pos = 'center'
        hor_pos = 'right'
    elif reverse == 'bottom':
        y_outer = outer
        ver_pos = 'bottom'
    elif reverse == False:
        y_outer = -outer

    return x_outer, y_outer, ver_pos, hor_pos


def extract_parameters(global_info):
    para_names = ['max_height', 'n_sample', 'delta_x', 'delta_y',
                  'stride', 'multi_y', 'x_lim', 'y_lim', 'x_max', 'y_max']
    for para_name in para_names:
        string = para_name + ' = global_info[\'' + para_name + '\']'
        exec(string, {'para_name': para_name, 'global_info': global_info}, globals())

    return max_height, n_sample, delta_x, delta_y, stride, multi_y, x_lim, y_lim, x_max, y_max


# STEP 1
def nj_Method(distance_df, start_id):
    info_dict_ = OrderedDict()
    num_dict_ = defaultdict(default_one_factory)
    while distance_df.shape[0] > 1:
        n = len(distance_df)
        if n > 2:
            matrix_Q, sum_mat = matrix_Q_compute(distance_df)
            min_num = np.argmin(matrix_Q)
            node1_iloc = min_num // n
            node2_iloc = min_num % n
            node1 = distance_df.columns[node1_iloc]
            node2 = distance_df.columns[node2_iloc]

            dist_node12 = distance_df.loc[node1, node2]
            dist_new_node1_ = 0.5 * (dist_node12 + (sum_mat[node1_iloc] - sum_mat[node2_iloc]) / (n - 2))
            dist_new_node2_ = dist_node12 - dist_new_node1_

        else:
            node1 = distance_df.columns[0]
            node2 = distance_df.columns[1]
            dist_node12 = distance_df.loc[node1, node2]
            dist_new_node1_ = 0.5 * distance_df.iloc[0, 1]
            dist_new_node2_ = dist_new_node1_

        count = 0
        if node1 in info_dict_.keys():
            count += info_dict_[node1][1]
        else:
            count += 1
        if node2 in info_dict_.keys():
            count += info_dict_[node2][1]
        else:
            count += 1
        info_dict_[start_id] = [{node1: dist_new_node1_, node2: dist_new_node2_}, count]
        num_dict_[start_id] = count

        distance_df = update_distance_df(distance_df, node1, node2, dist_node12, start_id)
        start_id += 1

    return info_dict_, num_dict_


# STEP 2
def calculate_tree_structure(info_dict_, num_dict_, n_sample, reverse,
                             x_lim, y_lim, x_max, y_max):
    exist_node_dict = {}
    NJ_tree_info = namedtuple('NJ_tree', ['global_info', 'position', 'structure', 'num_dict'])
    NJ_tree_info.structure = info_dict_.copy()
    NJ_tree_info.num_dict = num_dict_

    delta_x, delta_y, x_lim, y_lim, x_max, y_max = compute_delta(reverse, x_lim, y_lim, x_max,  y_max)

    # top branch
    info = info_dict_.popitem()

    node_info = info[1][0]
    whole_num = info[1][1]
    node1, node2 = list(node_info.keys())
    num1, num2 = num_dict_[node1], num_dict_[node2]
    x1 = num1 / 2 - 0.5
    x2 = num2 / 2 + num1 - 0.5
    y1 = - node_info[node1]
    y2 = - node_info[node2]
    exist_node_dict[node1] = [x1, y1, 'left', 0]
    exist_node_dict[node2] = [x2, y2, 'right', 0]

    # compute height of the tree
    info_dict__ = info_dict_.copy()
    exist_node_dict_copy = exist_node_dict.copy()
    while info_dict__:
        info_ = info_dict__.popitem()
        x1_, x2_, y1_, y2_, node1_, node2_, y_start_ = core_process(info_, exist_node_dict_copy, num_dict_, sort=False)
        exist_node_dict_copy[node1_] = [x1_, y1_, 'left', y_start_]
        exist_node_dict_copy[node2_] = [x2_, y2_, 'right', y_start_]
    tree_info = pd.DataFrame(exist_node_dict_copy.values())
    max_height = - np.min(tree_info.iloc[:, 1])

    # set tree size
    stride = delta_x / n_sample
    multi_y = delta_y / max_height
    NJ_tree_info.global_info = {'max_height': max_height, 'n_sample': n_sample,
                                'delta_x': delta_x, 'delta_y': delta_y,
                                'stride': stride, 'multi_y': multi_y,
                                'x_lim': x_lim, 'y_lim': y_lim, 'x_max': x_max, 'y_max': y_max}

    # rest subtree
    while info_dict_:
        info = info_dict_.popitem()
        x1, x2, y1, y2, node1, node2, y_start = core_process(info, exist_node_dict, num_dict_, sort=False)
        exist_node_dict[node1] = [x1, y1, 'left', y_start]
        exist_node_dict[node2] = [x2, y2, 'right', y_start]

    NJ_tree_info.position = exist_node_dict

    return NJ_tree_info


# STEP 3
def plot_tree(plt_, NJ_tree_info, colors_, reverse=False, x_lim=False, y_lim=False, x_max=False, y_max=False,
              show_node=False, parent_color='red', child_color='orange', scatter_size=50, marker='D', **kwargs):
    # extract parameters from NJ_tree_info
    info_dict_ = NJ_tree_info.structure.copy()
    num_dict_ = NJ_tree_info.num_dict
    exist_node_dict = NJ_tree_info.position
    global_info = NJ_tree_info.global_info

    max_height, n_sample, delta_x, delta_y, \
    stride, multi_y, x_lim_raw, y_lim_raw, x_max_raw, y_max_raw = extract_parameters(global_info)

    # update info
    if x_lim or y_lim or x_max or y_max:
        _update = True
    else:
        _update = False

    if not x_lim:
        x_lim = x_lim_raw
    if not y_lim:
        y_lim = y_lim_raw
    if not x_max:
        x_max = x_max_raw
    if not y_max:
        y_max = y_max_raw

    if _update:
        delta_x, delta_y, x_lim, y_lim, x_max, y_max = compute_delta(reverse, x_lim, y_lim, x_max, y_max)
        stride = delta_x / n_sample
        multi_y = delta_y / max_height
        NJ_tree_info.global_info = {'max_height': max_height, 'n_sample': n_sample,
                                    'delta_x': delta_x, 'delta_y': delta_y,
                                    'stride': stride, 'multi_y': multi_y,
                                    'x_lim': x_lim, 'y_lim': y_lim, 'x_max': x_max, 'y_max': y_max}

    # iterable plotting
    color_order = 0
    while info_dict_:
        info = info_dict_.popitem()
        node1, node2 = list(info[1][0].keys())
        x1, y1, strand1, y_start = exist_node_dict[node1]
        x2, y2, strand2, y_start = exist_node_dict[node2]

        x_ = np.array([x1, x1, x2, x2]) * stride + x_lim
        y_ = (np.array([y1, y_start, y_start, y2]) + max_height) * multi_y + y_lim

        node_color = []
        if show_node:
            if node1 in info_dict_.keys():
                node_color += [parent_color]
            else:
                node_color += [child_color]
            if node2 in info_dict_.keys():
                node_color += [parent_color]
            else:
                node_color += [child_color]

        plt_sub_tree(plt_, x_, y_, colors_[color_order], reverse, y_lim, y_max,
                     show_node=show_node, color=node_color,
                     scatter_size=scatter_size, marker=marker, **kwargs)

        color_order += 1

    return plt_, NJ_tree_info


# STEP 4
def label_taxa(plt_, info_only, label, label_rotation, reverse, label_fontcolor, label_fontsize,
               NJ_tree_info, compression_exponent=0.7, compression_divisor=9):
    if not label or info_only:
        return

    # extract parameters from NJ_tree_info
    global_info = NJ_tree_info.global_info
    exist_node_dict = NJ_tree_info.position
    max_height, n_sample, delta_x, delta_y, \
    stride, multi_y, x_lim, y_lim, x_max, y_max = extract_parameters(global_info)

    # get taxa nodes id
    keys = np.array(list(exist_node_dict.keys()))
    taxa_nodes = keys[keys < n_sample]

    # genarate text position parameters
    outer = delta_y / n_sample ** compression_exponent / compression_divisor
    x_outer, y_outer, ver_pos, hor_pos = text_pos_tune(reverse, outer)

    for node in taxa_nodes:
        x1, y1, strand, _ = exist_node_dict[node]

        x1 = x1 * stride + x_lim
        y1 = (y1 + max_height) * multi_y + y_lim

        if reverse == 'left':
            x1, y1 = mirror(y1, y_lim, y_max), x1
        elif reverse == 'right':
            x1, y1 = y1, x1
        elif reverse == 'bottom':
            x1, y1 = x1, mirror(y1, y_lim, y_max)

        plt_.text(x1 + x_outer, y1 + y_outer, str(label[node]), color=label_fontcolor, fontsize=label_fontsize,
                  rotation=label_rotation, verticalalignment=ver_pos, horizontalalignment=hor_pos)


def neibour_join_Tree(plt_, sample_in_column_vector=None, distance=None, info_only=False,
                      reverse=False, c=None, cmod=None, c_alpha_min=False,
                      label=False, label_rotation=0, label_fontsize=12, label_fontcolor='black',
                      x_lim=0, x_max=10, y_lim=0, y_max=6, return_structure=False,
                      show_node=False, parent_color='red', child_color='orange', scatter_size=50, marker='D', **kwargs):
    """
    Use neibour join methods to construct a nj-tree.
    :param plt_: plot canvas or subplot canvas obj
    :param sample_in_column_vector: numpy.array object with shape in (k, n), n is the sample size
    :param distance: distance matrix in numpy.array or pandas.core.frame.DataFrame form. If 'distance' is given,
    'sample_in_column_vector' will be neglect.
    :param info_only: if True, only return tree info.
    :param reverse: reverse tree plot, 'left', 'right', 'bottom' and False is allowed. default is False
    :param c: To identify the name of single color, cmap or discrete multiple colors.
    With using c and cmod, the function can understand the right color information.
	Str and list fill with str are both accepted. If str is input, all lines will use the same color name.
    :param cmod: To identify the color style to make function to understand the color name.
    int ot list with int (only 1,2 or 3) are allowed as the element.
    1 refer to type of color, 2 refer to type of cmap and 3 refer to discrete multiple colors.
    :param c_alpha_min: default is 1. which statement the gradient range of color transparency. 1 means no gradient.
    :param label: list of string, to label taxa nodes.
    :param label_rotation: rotation of string
    :param label_fontsize: fontsize of string
    :param label_fontcolor: fontcolor of string
    :param x_lim: minimum X-axis of the tree
    :param x_max: maximum X-axis of the tree
    :param y_lim: minimum Y-axis of the tree
    :param y_max: maximum Y-axis of the tree
    :param return_structure: return tree info.
    :param show_node: show each node of the tree
    :param parent_color: color for nodes belong to parent set
    :param child_color: color for nodes belong to child set
    :param scatter_size: node size
    :param marker: node marker
    :param kwargs: kwargs for plt.plot()
    :return: object
    """
    if sample_in_column_vector is None and distance is None:
        raise TypeError('neibour_join_Tree() missing column vector data or distance matrix.')
    _check_cmod(c, cmod)  # to ensure cmod is in [1, 2, 3]

    if distance is None:
        distance = compute_squared_EDM_method(sample_in_column_vector)

    n_sample = distance.shape[0]

    if type(distance) == pd.core.frame.DataFrame and label == True:
        label = list(distance.index)
        distance_df = distance
    else:
        distance_df = pd.DataFrame(distance)
        distance_df.columns = list(range(n_sample))
        distance_df.index = list(range(n_sample))

    ### NJ method
    # STEP1: nj Method
    info_dict_, num_dict_ = nj_Method(distance_df, n_sample)

    ### plot part
    colors_ = measure_color(c, cmod, c_alpha_min, n_sample)

    # STEP2: calculate tree structure
    NJ_tree_info = calculate_tree_structure(info_dict_, num_dict_, n_sample, reverse, x_lim, y_lim, x_max, y_max)

    # STEP3: plot tree
    if not info_only:
        plt_, NJ_tree_info = plot_tree(plt_, NJ_tree_info, colors_, reverse=reverse,
                                       x_lim=x_lim, y_lim=y_lim, x_max=x_max, y_max=y_max,
                                       show_node=show_node, parent_color=parent_color, child_color=child_color,
                                       scatter_size=scatter_size, marker=marker, **kwargs)

    # STEP4: label
    label_taxa(plt_, info_only, label, label_rotation, reverse, label_fontcolor, label_fontsize, NJ_tree_info,
               compression_exponent=0.7, compression_divisor=9)

    if info_only:
        return NJ_tree_info

    elif return_structure:
        return plt_, NJ_tree_info

    else:
        return plt_


def _tracing_ancestor(node_, parent_child_dict_):
    ancestor = []
    while node_ in parent_child_dict_.keys():
        parent = parent_child_dict_[node_]
        ancestor += [parent]
        node_ = parent

    return ancestor[::-1]


def _tracing_posterity(node_, info_dict_):
    under_find = [node_]
    posterity = []
    while under_find:
        node = under_find.pop()
        posterity += [node]
        candidate = list(info_dict_[node][0].keys())
        under_find += [node for node in candidate if node in info_dict_.keys()]

    posterity.sort()
    sub_info_dict_ = OrderedDict()
    for node in posterity:
        sub_info_dict_[node] = info_dict_[node]

    return sub_info_dict_


def sub_Tree_plot(plt_, NJ_tree_info, contain_nodes_list=None, contain_label_list=None,
                  info_only=False, reverse=False, c=False, cmod=False, c_alpha_min=False,
                  label=False, label_rotation=0, label_fontsize=12, label_fontcolor='black',
                  x_lim=0, x_max=10, y_lim=0, y_max=6, return_structure=False,
                  show_node=False, parent_color='red', child_color='orange', scatter_size=50, marker='D', **kwargs):
    """
    Automatically draw the minimum subgraph containing specified terminal nodes.
    :param NJ_tree_info: NJ_tree_info returned by neibour_join_Tree()
    :param contain_nodes_list: containing these terminal nodes
    :param contain_label_list: containing these terminal labels
    other keyword arguments please see neibour_join_Tree()
    :return: object
    """
    if contain_nodes_list is None and contain_label_list is None:
        raise TypeError('sub_Tree_plot() missing contain-node ids or contain-taxa labels.')
    _check_cmod(cmod)  # to ensure cmod is in [1, 2, 3]

    # extract parameters from NJ_tree_info
    info_dict_ = NJ_tree_info.structure.copy()
    num_dict_ = NJ_tree_info.num_dict

    ### update subtree info
    parent_child_dict = {}
    for key, value in zip(info_dict_.keys(), info_dict_.values()):
        value = list(value[0].keys())
        parent_child_dict[value[0]] = key
        parent_child_dict[value[1]] = key

    if contain_label_list is not None:
        contain_nodes_list = [np.argmax(np.array(label) == lab) for lab in contain_label_list]

    nodes_ancestor = np.array(pd.DataFrame([_tracing_ancestor(node, parent_child_dict) for node in contain_nodes_list]))
    same_ancestor_array = np.mean(nodes_ancestor / nodes_ancestor[0] == 1, axis=0)
    the_node_order = len(same_ancestor_array) - np.argmax(same_ancestor_array[::-1] == 1) - 1
    start_from = int(nodes_ancestor[0][the_node_order])
    sub_info_dict = _tracing_posterity(start_from, info_dict_)
    n_sample = len(sub_info_dict.keys()) + 1

    ### plot part
    colors_ = measure_color(c, cmod, c_alpha_min, n_sample)

    # STEP2: calculate tree structure
    NJ_tree_info_new = calculate_tree_structure(sub_info_dict, num_dict_, n_sample, reverse, x_lim, y_lim, x_max, y_max)

    # STEP3: plot tree
    if not info_only:
        plt_, NJ_tree_info_new = plot_tree(plt_, NJ_tree_info_new, colors_, reverse=reverse,
                                       x_lim=x_lim, y_lim=y_lim, x_max=x_max, y_max=y_max,
                                       show_node=show_node, parent_color=parent_color, child_color=child_color,
                                       scatter_size=scatter_size, marker=marker, **kwargs)

    # STEP4: label
    label_taxa(plt_, info_only, label, label_rotation, reverse, label_fontcolor, label_fontsize, NJ_tree_info_new,
               compression_exponent=0.7, compression_divisor=9)

    if info_only and return_structure:
        return NJ_tree_info_new

    elif return_structure:
        return plt_, NJ_tree_info_new

    else:
        return plt_


if __name__ == "__main__":
    import zsx_some_tools as st

    dist_df = st.read_file('C:/Users/kevin/Desktop/python/张森欣/experiment_fold/phylogenetic_plot/distance.txt',
                           index_col=0)

    plt.figure(figsize=(10, 10))
    tree_, NJ_tree_info = neibour_join_Tree(plt, distance=dist_df, reverse='left',
                                               c='YlOrBr', cmod=2, c_alpha_min=0.3, label=list(dist_df.index),
                                               return_structure=True, show_node=True,
                                               parent_color='red', child_color='orange', scatter_size=30, marker='D')
    plt.show()





