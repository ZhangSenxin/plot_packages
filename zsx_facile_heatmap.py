import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections.abc import Iterable


def facile_heatmap(ax, data_in, *save, c=None, cmod=False, cr=True, width=1, height=1, rotation=0, zorder=1,
                   title=None, title_size=20, show_values=False, raw_value=True, round_num=3,
                   interval_x=0.5, interval_y=0.5, edgecolor=None, linewidth=0, size_c=10, size_r=10,
                   divide_alpha=1, rotation_x=0, rotation_y=0, fontsize=15, x0=0, y0=0, canvas=True, shape='square',
                   fig_x=10, fig_y=10, axis='off', col_label=None, ind_label=None, islabel=True, font_color='black',
                   fontproperty_x=None, fontproperty_y=None,
                   normalize=False, ignore_warning=False, **color_dict_name):
    """

    :param ax: To determine weather plt or subplt.
    :param data_in: Should be a dataframe filled with int or float.
    :param save: To save your figure, you should give two paras, which are "save_path” and “ file_name".
    :param c: To identify the name of color, cmap or dictionary.
    With using 'c' and 'cmod', the function can understand the right color information.
	Str and list fill with str are both accepted.
	If str is input, all block will use the same color name.
	Besides, list input can cooperate with parameter cr to determine whether color is different among rows or columns.
	If list input, the length of c should as long as the length of data's row or column, which of it is determined by cr.
	Default will use color:
	['red', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'fuchsia'] with repeating, and cmod=1.
    :param cmod: To identify the color style to make function to understand the color name.
    int ot list with int (only 1,2 or 3) are allowed as the element.
    1 refer to type of color, 2 refer to type of cmap and 3  refer to type of dictionary.
	Type 1(color): Solid color with different alpha determined by data's value.
	Type 2(cmap): Color from different location of the colorbar.
	Type 3(dictionary): Color will be determined by the dictionary that map values to colors,
	the color dictionary is defined by yourself and should input in the keyword argument color_dict_name.
	If c is list of names, the length of cmod should as long as the length of c.
    :param cr: 	Default set different colours among different rows.
    If cr = True, different colours among different columns.
    :param color_dict_name:	Input color dictionary you used in c.
    The format should be dict_name=dict_name.
    For example, if you define a dictionary called 'color_dict1',
    and use it in parameter c, youshould give the keyword argument that: color_dict1=color_dict1
    :param width: grid width.
    :param height: grid height.
    :param rotation: rotate whole heatmap in degrees
    :param title: Give title.
    :param title_size: title fontsize
    :param show_values: Show values in each block or not.
    :param raw_value: Show input data values or normalize data values.
    :param round_num: n for round(value, n)
    :param interval_x: Interval between x label and heatmap.
    :param interval_y: Interval between y label and heatmap.
    :param edgecolor: edge color for each block
    :param linewidth: linewidth for edge
    :param size_c: Size of columns' label.
    :param size_r: Size of rows' label.
    :param divide_alpha: To adjust the alpha of all blocks.
    :param rotation_x: Rotation of x labels.
    :param rotation_y: Rotation of y labels.
    :param fontsize: Font size.
    :param x0: heatmap pos x
    :param y0: heatmap pos y
    :param canvas: if make canvas automaticaly
    :param shape: if fit shape size to make each block as 'square'
    :param fig_x: Width of the figure.
    :param fig_y: Height of the figure.
    :param axis: Show axis or not.
    :param col_label: Column label
    :param ind_label: Index label
    :param islabel: Show label or not
    :param font_color: Font color for each block
    :param fontproperty_x: Font property
    :param fontproperty_y:
    :param normalize: Normalize value, 'all', 'col', and 'row' is allowed.
    :param ignore_warning: Ignore this function's warning.


    :return: ax object
    """
    if cmod and type(cmod) != list:
        cmod = [cmod]
    if c is not None and type(c) != list:
        c = [c]

    # 处理报错信息
    if c is not None and not cmod:
        raise ValueError('Error! Parameter \'cmod\' is missing!')
    elif cmod:
        try:
            if (set(cmod) | {1, 2, 3}) != {1, 2, 3}:
                raise ValueError('Error! Element in parameter \'cmod\' is Illegal! Please use 1,2 or 3 to fill it.')
        except TypeError:
            if not (cmod in [1, 2, 3]):
                raise ValueError('Error! Element in parameter \'cmod\' is Illegal! Please use 1,2 or 3 to fill it.')
    if c is not None:
        if len(c) != len(cmod):
            raise ValueError('Error! Length of parameter \'c\' and parameter \'cmod\' are not matched!')
    if cmod:
        if 3 in cmod:
            color_dict_name_1 = np.array(c)[np.array(cmod) == 3]
            color_dict_name_2 = [i for i in color_dict_name.keys()]
            if len(set(color_dict_name_1) - set(color_dict_name_2)) != 0:
                print(color_dict_name_2)
                raise ValueError('Error! parameter \'color_dict_name\' can not '
                                 'match to dictionaries in parameter \'c\'')

    data = data_in.copy()

    if type(data) == pd.core.frame.DataFrame:
        if col_label is None:
            col_label = list(data.columns)
        if ind_label is None:
            ind_label = list(data.index)
        data = np.array(data, dtype=float)

    # 提取大小参数信息
    rows = data.shape[0]
    columns = data.shape[1]

    # 自适应大小，且防止像素过多报错
    if canvas:
        xx = 2 * columns + interval_x + 0.5
        yy = 2 * rows + interval_y + 0.5
        r = yy/xx
        if shape == 'square':
            if r < 1:
                plt.figure(figsize=(16, 16 * r))
            else:
                plt.figure(figsize=(16 / r, 16))
        else:
            plt.figure(figsize=(fig_x, fig_y))

    # 处理默认颜色
    fast = 0
    if c is None:
        # c = ['red', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'fuchsia'] * (max(data.shape) // 7 + 1)
        c = ['red'] * max(data.shape)
        cmod = [1] * max(data.shape)

    else:
        if len(c) == 1 and (3 not in cmod):
            fast = 1
        elif cr and len(c) < rows:
            c = list(c) * (data.shape[0] // len(c) + 1)
        elif not cr and len(c) < columns:
            c = list(c) * (data.shape[1] // len(c) + 1)

        if fast:
            cmod = cmod[0]
            c = c[0]
        else:
            if len(cmod) == 1:
                cmod = cmod * max(data.shape)
            elif cr and len(cmod) < rows:
                cmod = list(cmod) * (data.shape[0] // len(cmod) + 1)
            elif not cr and len(cmod) < columns:
                cmod = list(cmod) * (data.shape[1] // len(cmod) + 1)

    # 确定子图与主图
    if ax == plt:
        ax = ax.gca()

    # if rotation:
    #     max_length = max(columns, rows)
    #     expand = max_length * (2 ** 0.5 - 1) / 2
    # else:
    #     expand = 0

    # ax.set_xlim(-interval_x + x0 - expand, columns + interval_x + x0 + expand)
    # ax.set_ylim(-interval_y + y0 - expand, rows + interval_y + y0 + expand)

    # 处理数据归一化（全局、按行、按列）
    # all, column, row

    if not normalize:
        _do = 0  # 若cmod为3则不强制标准化
        if type(cmod) == int:
            if cmod != 3:
                _do = 1
        else:
            if 3 not in cmod:
                _do = 1
        if _do == 1:
            min_value = np.min(np.min(data))
            if min_value < 0:
                data -= min_value
                if not ignore_warning:
                    print("Warning! Values in DataFrame may lower than zero, values are added by min_value.")
            max_value = np.max(np.max(data))
            if max_value > 1:
                data /= max_value
                if not ignore_warning:
                    print("Warning! Values in DataFrame may higher than one, values are divided by max_value.")

    elif normalize in 'all':
        min_value = np.min(np.min(data))
        data -= min_value
        max_value = np.max(np.max(data))
        data /= max_value

    elif normalize in 'row':
        for i in range(rows):
            min_value = np.min(data[i, :])
            data[i, :] = data[i, :] - min_value
            max_value = np.max(data[i, :])
            data[i, :] = data[i, :] / max_value

    elif normalize in 'column':
        for i in range(columns):
            min_value = np.min(data[:, i])
            data[:, i] = data[:, i] - min_value
            max_value = np.max(data[:, i])
            data[:, i] = data[:, i] / max_value

    x, y = np.meshgrid(np.array(range(columns + 1)), np.array(range(rows + 1)))
    x = x * width + x0
    y = y * height + y0

    center_x = columns * width / 2 + x0
    center_y = rows * height / 2 + x0
    tr = transforms.Affine2D().rotate_deg_around(center_x, center_y, rotation)

    if fast:
        trans_data = data[::-1, :].copy()
        trans_data = trans_data / divide_alpha
        if cmod == 1:
            cmap1 = LinearSegmentedColormap.from_list("1", ['white', c])
        else:
            cmap1 = c

        ax.pcolormesh(x, y, trans_data, cmap=cmap1, vmin=0, vmax=1, zorder=1,
                      edgecolor=edgecolor, linewidth=linewidth, shading='flat',
                      transform=tr + ax.transData)

    else:
        # 计算每个block的颜色
        colors_map_ = []
        if cr:
            iter_ = range(rows)
        else:
            iter_ = range(columns)

        for i_ in iter_:
            color_ = c[i_]
            mod_ = cmod[i_]
            if cr:
                da_ = data[i_, :]
            else:
                da_ = data[:, i_]

            if mod_ == 1:
                da_ = da_ / divide_alpha
                cmap1 = LinearSegmentedColormap.from_list("1", ['white', color_])
                colors_ = [cmap1(d_ - 1e-6) for d_ in da_]

            elif mod_ == 2:
                da_ = da_ / divide_alpha
                try:
                    cmap1 = cm.get_cmap(color_)
                except ValueError:
                    cmap1 = color_
                colors_ = [cmap1(d_ - 1e-6) for d_ in da_]

            else:
                color_dict_ = color_dict_name[color_]
                colors_ = [color_dict_[d_] for d_ in da_]

            colors_map_ += [colors_]

        if not cr:
            colors_map_ = [[colors_map_[i][j] for i in range(columns)] for j in range(rows)]

        # 将所有颜色集合重新给定id
        for i in range(rows):
            for j in range(columns):
                if not isinstance(colors_map_[i][j], str):
                    colors_map_[i][j] = tuple(colors_map_[i][j])
        color_sets_ = list(set([i for k in colors_map_ for i in k]))
        cmap_all_ = ListedColormap(color_sets_)

        color_dict_id = {}
        for i_, c_ in enumerate(color_sets_):
            color_dict_id[c_] = i_ / (len(color_sets_) - 1)

        # 根据颜色id给数据矩阵重新赋值，按行倒置使得热图与数据框形式一致
        trans_data = pd.DataFrame([[color_dict_id[c_] for c_ in colors_map_[i]] for i in range(rows)])
        trans_data = trans_data.iloc[::-1, :]

        # 按照转换后的数据绘绘制热图
        ax.pcolormesh(x, y, trans_data, cmap=cmap_all_, vmin=0, vmax=1, zorder=1,
                      edgecolor=edgecolor, linewidth=linewidth, shading='flat',
                      transform=tr + ax.transData)

    if show_values:
        size_fold = 1
        if raw_value:
            value_label = np.round(np.array(data_in, dtype=float), round_num)
        else:
            value_label = np.round(data, round_num)

        len_value = [len(str(value_label[i, j])) for i in range(rows) for j in range(columns)]

        if max(len_value) >= 3:
            size_fold = max(len_value) / 3
        for i in range(rows):
            for j in range(columns):
                ax.text((j + 0.5) * width + x0, (rows - i - 0.5) * height + y0, value_label[i, j], color=font_color,
                        fontsize=fontsize/size_fold, verticalalignment="center", horizontalalignment="center")

    # 标注label
    if islabel:
        if col_label is None:
            col_label = range(columns)
        if ind_label is None:
            ind_label = range(rows)

        for i in range(columns):
            ax.text((i + 0.5) * width + x0, (rows + 0.6 * interval_y) * height + y0, col_label[i],
                    rotation=rotation_x, color=font_color, rotation_mode='anchor', fontsize=size_c,
                    verticalalignment="center", horizontalalignment="center", fontproperties=fontproperty_x)
        for i in range(rows):
            ax.text((-0.6 * interval_x) * width + x0, (rows - i - 0.5) * height + y0, ind_label[i],
                    rotation=rotation_y, color=font_color, rotation_mode='anchor', fontsize=size_r,
                    verticalalignment="center", horizontalalignment="center", fontproperties=fontproperty_y)

    if title is not None:
        ax.set_title(title, fontsize=title_size, color=font_color)

    ax.axis(axis)

    try:
        plt.savefig(save[0] + save[1], bbox_inches='tight')
        if not ignore_warning:
            print("Fig has been saved.")
    except IndexError:
        pass

    return ax


def cluster_facile_heatmap(ax, data_in, cluster_col=True, cluster_row=True, metric='euclidean', method='ward',
                           cluster_num=3, height_threshold=None, cluster_color_list=False, cluster_bgcolor='black',
                           cluster_label=False, cluster_dict=False, metric_y='euclidean', method_y='ward',
                           cluster_num_y=3, height_threshold_y=None, cluster_color_list_y=False,
                           cluster_bgcolor_y='black',
                           cluster_label_y=False, cluster_dict_y=False, cluster_height_x=1, cluster_height_y=2,
                           cluster_interval_x=0.3, cluster_interval_y=0.3, x0=0, y0=0, return_data=False,
                           *save, **kwargs):

    try:
        import zsx_hierarchical_clustering as hc
    except ModuleNotFoundError:
        print('Module :', 'zsx_hierarchical_clustering', 'is not found')
        return None

    if ax == plt:
        ax = ax.gca()

    if cluster_col:
        if not cluster_dict:
            cluster_dict = {}
        coordinate1 = hc.hierarchical_plot(ax, data_in.T, metric=metric, method=method, cluster_num=cluster_num,
                                           height_threshold=height_threshold, reverse='top',
                                           color_list=cluster_color_list,
                                           label=cluster_label,
                                           x0=x0 + 0.5,
                                           x1=x0 + 0.5 + data_in.shape[1],
                                           y0=y0 + cluster_interval_y + data_in.shape[0],
                                           y1=y0 + cluster_interval_y + data_in.shape[0] + cluster_height_y,
                                           bgcolor=cluster_bgcolor, return_coordinate=True, **cluster_dict)

    if cluster_row:
        if not cluster_dict_y:
            cluster_dict_y = {}
        coordinate2 = hc.hierarchical_plot(ax, data_in, metric=metric_y, method=method_y, cluster_num=cluster_num_y,
                                           height_threshold=height_threshold_y, reverse='left',
                                           color_list=cluster_color_list_y,
                                           label=cluster_label_y,
                                           x0=x0 - cluster_interval_x - cluster_height_x,
                                           x1=x0 - cluster_interval_x,
                                           y0=y0 + 0.5,
                                           y1=y0 + 0.5 + data_in.shape[0],
                                           bgcolor=cluster_bgcolor_y, return_coordinate=True, **cluster_dict_y)
    if cluster_col:
        col_order = pd.DataFrame(coordinate1).sort_values(by=1).iloc[:, 0]
    else:
        col_order = list(range(data_in.shape[1]))

    if cluster_row:
        row_order = pd.DataFrame(coordinate2).sort_values(by=1).iloc[:, 0]
    else:
        row_order = list(range(data_in.shape[0]))

    data_in2 = data_in.iloc[row_order, col_order]

    facile_heatmap(ax, data_in2, x0=x0, y0=y0, canvas=False, *save, **kwargs)


    # ax.set_xlim(x0 - 0.3 - cluster_height_x - cluster_interval_x - 0.5, x0 + 0.5 + data_in.shape[1] + 0.5)
    # ax.set_ylim(-0.5 + y0, y0 + 0.3 + data_in.shape[0] + cluster_height_y + cluster_interval_y + 0.5)

    if return_data:
        return ax, data_in2
    else:
        return ax


if __name__ == "__main__":
    data = pd.DataFrame(np.random.random(20).reshape(4, 5))

    facile_heatmap(plt, data)
    plt.show()

