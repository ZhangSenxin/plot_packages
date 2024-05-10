#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

import zsx_some_tools as st
import zsx_facile_heatmap as fh
import zsx_color_bar as cb
import zsx_HMM as hmm


class HicHeatmapPlotting(object):
    """
    Framework for Hic matrix plotting, contain insulation analysis.
    Sometimes we can perfrom with other type of data, which is in square matrix form.
    """
    def __init__(self, mat_norm_, mat_raw_, chr_start_, chr_end_, res_,
                 tickts_=None, bar_label_list_=None, mat_unbalanced=None):
        """
        initialize and compute global parameters
        :param mat_norm_: normalized matrix for main heatmap plotting
        :param mat_raw_: raw matrix for insulation analysis
        :param chr_start_: start position, which influence the xtickts
        :param chr_end_: end position, which influence the xtickts
        :param res_: resolution to compute how many bins
        :param tickts_: main heatmap colorbar's tickts, range in [0, 1], relative position
        :param bar_label_list_: main heatmap colorbar's tickts label, None is allowed
        :param mat_unbalanced: unbalanced matrix, each value indicates read counts
        """
        self.cmap1 = LinearSegmentedColormap.from_list("R", ['white', 'r'])
        self.mat_ = mat_norm_
        self.res_ = res_
        self.mat_raw_ = mat_raw_
        self.tickts = tickts_
        self.bar_label_list = bar_label_list_
        self.mat_unbalanced = mat_unbalanced

        center_x = mat_norm_.shape[1] / 2
        center_y = mat_norm_.shape[0] / 2

        self.center_x = center_x
        self.center_y = center_y

        interval = 0.05 * center_y
        y_use = center_y - interval
        x_start, x_end = self.trans_x_value(0, center_x), self.trans_x_value(mat_norm_.shape[0], center_x)

        self.interval = interval
        self.y_use = y_use
        self.x_start = x_start
        self.x_end = x_end

        # compute label info
        chr_start_pos = int(chr_start_.replace(',', ''))
        chr_end_pos = int(chr_end_.replace(',', ''))
        chr_start_pos = chr_start_pos // res_ * res_
        chr_end_pos = (chr_end_pos // res_ + 1) * res_ if chr_end_pos % res_ != 0 else (chr_end_pos // res_) * res_
        bins = (chr_end_pos - chr_start_pos) // res_
        start_stride = 5
        label_num = bins // start_stride
        while label_num > 4:
            start_stride += 5
            label_num = bins // start_stride
        label_bin = [start_stride * i_ for i_ in range(label_num + 1)]
        if bins // start_stride != 0:
            label_bin = label_bin[:-1] + [bins]
        label_name = [reform_genome_pos(chr_start_pos + i_ * res_, rounding=3) for i_ in label_bin]

        self.bins = bins
        self.label_bin = label_bin
        self.label_name = label_name
        self.chr_start_pos = chr_start_pos
        self.chr_end_pos = chr_end_pos

        self.length = None
        self.cbar_x0 = None
        self.cbar_x = None
        self.y0 = None
        self.figure_height_range = None
        self.legend_x = None
        self.legend_y = None
        self.k = None

    @staticmethod
    def trans_x_value(value_, center_x_):
        return center_x_ + (value_ - center_x_) * (2 ** 0.5)

    def get_insulation_score(self, mat1_raw_, window_s, up_down='up', ignore_bins=1, norm=True):
        """
        Insulation mathod
        :param mat1_raw_: raw matrix (un-log-normed)
        :param window_s: window list for inputation analysis, values indicate bins
        :param up_down: plot at 'up' of main heatmap or 'down'
        :param ignore_bins: ignore how many bins close to main diagonal of a matrix
        :return: log2-normed-insulation df
        """
        if type(mat1_raw_) != np.ndarray:
            mat1_raw_ = np.array(mat1_raw_)

        all_insulation_scores = []
        for window in window_s:
            insulation_scores = []
            for i in range(mat1_raw_.shape[0]):
                if i < window or i >= mat1_raw_.shape[0] - window:
                    insulation_scores += [-1]
                    continue

                sub_mat = mat1_raw_[i - window: i + window + 1, i - window: i + window + 1]
                nans_ = np.isnan(sub_mat)
                sub_mat[nans_] = 0

                if up_down in 'upper':
                    sub_mat_tri = np.triu(sub_mat, k=ignore_bins + 1)  # up triangle matrix
                elif up_down in 'down':
                    sub_mat_tri = np.tril(sub_mat, k=-(ignore_bins + 1))  # down triangle matrix
                insulation_score = np.sum(sub_mat_tri)
                insulation_scores += [insulation_score]

            all_insulation_scores += [insulation_scores]

        if norm:
            all_insulation_scores_norm = self.norm_insulation_matrix(all_insulation_scores, window_s)
            return all_insulation_scores_norm
        else:
            # all_insulation_scores = pd.DataFrame(all_insulation_scores)
            return all_insulation_scores

    @staticmethod
    def norm_insulation_matrix(all_insulation_scores, window_s):
        """
        Norm function
        :param all_insulation_scores: insulation df
        :param window_s: window list for inputation analysis, values indicate bins
        :return:
        """
        all_insulation_scores = pd.DataFrame(all_insulation_scores, index=window_s)
        all_insulation_scores = all_insulation_scores.T

        mean = np.mean(all_insulation_scores[all_insulation_scores != -1], axis=0)
        for i in range(len(window_s)):
            all_insulation_scores.loc[all_insulation_scores.iloc[:, i] == -1,
                                      all_insulation_scores.columns[i]] = mean.iloc[i]

        all_insulation_scores = np.log2(all_insulation_scores / mean)
        all_insulation_scores_norm = st.double_norm(all_insulation_scores.T, axis=1)

        return all_insulation_scores_norm

    def plot(self, windows=None, plot_style='line', view_distance=False, plt_figure=True, **kwarg):
        """
        Perform one matrix's half-heatmap (rotate 45 degree) with inputation analysis plotting.
        :param windows: window list for inputation analysis, values indicate bins, default is [5, 6, 7, 8, 9, 10]
        :param plot_style: 'line', 'heatmap' and 'directionality' is allowed, control the plot style
        :param view_distance: only show interaction below this distance
        :return: None
        """
        x_start = self.x_start
        x_end = self.x_end
        center_x = self.center_x
        interval = self.interval
        bins = self.bins
        res_ = self.res_

        if windows is None:
            windows = [5, 6, 7, 8, 9]
        windows.sort()
        windows = windows[::-1]
        length = len(windows)

        ### compute figure size ###
        if not view_distance:
            view_bins = bins
        else:
            view_bins = view_distance // res_
        view_height = view_bins * 2 ** 0.5 / 2
        fig_height = bins * 2 ** 0.5 / 2

        height = int(abs(self.trans_x_value(0, center_x) - center_x) / 10)
        expand_height = 0
        if plot_style in 'heatmap':
            if length < 3:
                expand_height = 3 * interval + height * 3
            else:
                expand_height = 3 * interval + height * length

        elif plot_style in 'line' or plot_style in 'directionality':
            expand_height = 3 * interval + height * 4

        figure_height = view_height + expand_height
        figure_width = x_end - x_start + 11 * interval
        self.legend_x = (x_end - x_start + 7 * interval) / (x_end - x_start + 15 * interval)
        self.legend_y = 2 * interval / (figure_height + 2 * interval)

        k = (figure_height + 6 * interval) / (figure_width + 4 * interval)
        self.k = k

        ### create figure ###
        if plt_figure:
            plt.figure(figsize=(11, 11 * k))

        ### plot HicHeatmap ###
        self.hic_heatmap_plot(view_distance=view_distance)

        ### plot ana ###
        if plot_style in 'directionality':
            ### plot directionality ###
            self.hic_directionality_plot(windows)

        else:
            ### plot insulation ###
            fig_height_add = 4 if plot_style in 'line' else 3 if length < 4 else length
            self.hic_insulation_plot(windows, plot_style, fig_height_add, **kwarg)

        ### give lim ###
        fhr = self.figure_height_range
        plt.xlim(x_start - 2 * interval, x_end + 13 * interval)
        plt.ylim(fhr[0] - 4 * interval, fhr[1] + 2 * interval)

    def hic_heatmap_plot(self, tickts=None, bar_label_list=None, view_distance=False):
        """
        plot main heatmap (rotate 45 degree), with a isolate xtickts and a colorbar
        :param view_distance: only show interaction below this distance
        :return:
        """
        mat_ = self.mat_
        center_x = self.center_x
        center_y = self.center_y
        label_bin = self.label_bin
        label_name = self.label_name
        interval = self.interval
        y_use = self.y_use
        x_start = self.x_start
        x_end = self.x_end

        res_ = self.res_
        bins = self.bins

        if tickts is None:
            tickts = self.tickts
        if bar_label_list is None:
            bar_label_list = self.bar_label_list

        if not view_distance:
            view_bins = bins
        else:
            view_bins = view_distance // res_
        view_height = view_bins * 2 ** 0.5 / 2
        fig_height = bins * 2 ** 0.5 / 2

        self.figure_height_range = [center_y, center_y + view_height]

        # plot heatmap
        fh.facile_heatmap(plt, mat_, c='r', cmod=1, rotation=45, canvas=False, islabel=False, normalize='all')
        plt.fill_between([x_start, x_end], [center_y, center_y], [-center_y, -center_y], color='white')
        if view_height:
            plt.fill_between([x_start, x_end],
                             [center_y + view_height, center_y + view_height],
                             [center_y + fig_height, center_y + fig_height], color='white')

        # plot xticks
        plt.plot([x_start, x_end], [y_use, y_use], c='black')
        for i_, name_ in zip(label_bin, label_name):
            i_ = self.trans_x_value(i_, center_x)
            i_ = x_end if i_ > x_end else i_
            plt.plot([i_, i_], [y_use, y_use + 0.03 * center_y], c='black')
            plt.text(i_, y_use - 1.5 * interval, name_, horizontalalignment='center', fontsize=12)

        # hic color bar
        cbar_x0 = x_end + interval * 5
        cb_height = view_height
        cb.colorbar(plt, self.cmap1, cmod=2, height=cb_height, width=cb_height / 10,
                    x0=cbar_x0, y0=center_y,
                    outer=0.5, ticks=tickts, labels=bar_label_list, cr=False, lc='black')

        cbar_x = cbar_x0 + cb_height / 10
        for i in tickts:
            add = center_y
            plt.plot([cbar_x, cbar_x + interval / 2],
                     [i * cb_height + add, i * cb_height + add],
                     c='black', lw=1)

        self.cbar_x0 = cbar_x0
        self.cbar_x = cbar_x

    def hic_insulation_plot(self, windows, plot_style, fig_height_add, y0=None,
                            up_down='down', ignore_bins=1, legend=True, **kwarg):
        """
        plot insulation analysis,
        :param windows: window list for inputation analysis, values indicate bins, default is [5, 6, 7, 8, 9, 10]
        :param plot_style: 'line' and 'heatmap' is allowed, control the plot style
        :param fig_height_add: a number reletive to fold-change of parameter:height that insulation plot use
        :param y0: y start for insulation plot
        :param up_down: plot at 'up' of main heatmap or 'down'
        :param ignore_bins: ignore how many bins close to main diagonal of a matrix
        :param legend: plot line-plot legend or not
        :return:
        """
        center_x = self.center_x
        center_y = self.center_y
        interval = self.interval
        y_use = self.y_use
        x_start = self.x_start
        resolution = self.res_

        insulation_scores_norm = self.get_insulation_score(self.mat_raw_, window_s=windows,
                                                           up_down=up_down, ignore_bins=ignore_bins)
        length = len(windows)
        height = int(abs(self.trans_x_value(0, center_x) - center_x) / 10)

        ## plot insulation heatmap ##
        if plot_style == 'heatmap':
            cbar_x0 = self.cbar_x0
            cbar_x = self.cbar_x

            if length < 3:
                expand_height = 3 * interval + height * 3
            else:
                expand_height = 3 * interval + height * length
            if y0 is None:
                y0 = y_use - expand_height
                self.y0 = y0

            fh.facile_heatmap(plt, insulation_scores_norm, c='bwr', cmod=2, islabel=True, size_r=12,
                              x0=x_start, width=2**0.5, y0=y0, height=height,
                              col_label=[''] * insulation_scores_norm.shape[1],
                              ind_label=[reform_genome_pos(i * resolution) for i in windows],
                              interval_x=center_x // 7, normalize='all', canvas=False)

            # insulation color bar
            insulation_tickts = [0, 0.25, 0.5, 0.75, 1]
            insulation_label = [-1, -0.5, 0, 0.5, 1]
            insulation_height = height * max(len(windows), 3)
            cb.colorbar(plt, 'bwr', cmod=2, height=insulation_height, width=center_y / 10, x0=cbar_x0, y0=y0,
                        outer=0.5, ticks=insulation_tickts, labels=insulation_label, cr=False, lc='black')

            for i_ in insulation_tickts:
                plt.plot([cbar_x, cbar_x + interval / 2],
                         [i_ * insulation_height + y0, i_ * insulation_height + y0],
                         c='black', lw=1)

        ## plot insulation line ##
        elif plot_style == 'line':
            label_bin = self.label_bin
            label_name = self.label_name
            x_end = self.x_end

            insulation_line_x = [self.trans_x_value(i, center_x) for i in insulation_scores_norm.columns]

            expand_height = 3 * interval + height * 4
            if y0 is None:
                legend_set = True
                y0 = y_use - expand_height
                self.y0 = y0
            else:
                legend_set = False

            for i_, ind in enumerate(insulation_scores_norm.index[::-1]):
                lab = str(ind * 5) + ' kb'
                insulation_line_y = insulation_scores_norm.loc[ind, :] * height * 4 + y0
                plt.plot(insulation_line_x, insulation_line_y,
                         color=cm.Blues(1 - (i_ + 1) / (length * 1.5)),
                         label=lab, zorder=100 - i_, **kwarg)

            legend_start = 10 * interval / (x_start + x_end + 2 * interval)
            legend_height = fig_height_add / 15

            if legend:
                if up_down in 'upper':
                    plt.legend(loc='best', bbox_to_anchor=(legend_start, 1 - legend_height,
                                                           1 - legend_start, legend_height))
                elif up_down in 'down':
                    if not legend_set:
                        plt.legend(loc='best', bbox_to_anchor=(legend_start, 0,
                                                               1 - legend_start, legend_height))
                    else:
                        plt.legend(loc=[self.legend_x, self.legend_y])

            ytickt_x0 = insulation_line_x[0] - interval
            xtickt_y0 = y0 - interval

            # ytickt
            plt.plot([ytickt_x0, ytickt_x0],
                     [xtickt_y0, y0 + height * 4 + interval], color='black')
            for i_, lab in enumerate([-1, -0.5, 0, 0.5, 1]):
                y0_use = y0 + i_ * height
                plt.plot([ytickt_x0 - 1 * interval, ytickt_x0], [y0_use, y0_use], c='black')
                plt.text(ytickt_x0 - interval, y0_use, lab,
                         horizontalalignment='right', verticalalignment='center', fontsize=12)

            # xtickt
            plt.plot([ytickt_x0, insulation_line_x[-1] + interval],
                     [y0 - interval, y0 - interval], color='black')
            for i_, name in zip(label_bin, label_name):
                i_ = self.trans_x_value(i_, center_x)
                i_ = x_end if i_ > x_end else i_
                plt.plot([i_, i_], [xtickt_y0 - 0.6 * interval, xtickt_y0], c='black')
                plt.text(i_, xtickt_y0 - 1 * interval - interval, name,
                         horizontalalignment='center', fontsize=12)

        fhr = self.figure_height_range
        if fhr is not None:
            if up_down in 'upper':
                fhr[1] = fhr[1] + expand_height
            elif up_down in 'down':
                fhr[0] = fhr[0] - expand_height
            self.figure_height_range = fhr

    def plot_compare(self, windows=None, plot_style='line',
                     view_distance=100000, insulation_plot_pos=None, ignore_bins=1, **kwarg):
        """
        Perform compared matrix's heatmap (rotate 45 degree) with inputation analysis plotting.
        :param windows: window list for inputation analysis, values indicate bins, default is [5, 6, 7, 8, 9, 10]
        :param plot_style: 'line', 'heatmap' and 'directionality' is allowed, control the plot style
        :param view_distance: only show interaction below this distance
        :param insulation_plot_pos: 'upper', 'down', 'separate' and None is allowed,
                                    to control where to plot inputation analysis
        :param ignore_bins: ignore how many bins close to main diagonal of a matrix
        :return:
        """
        x_start = self.x_start
        x_end = self.x_end
        center_x = self.center_x
        interval = self.interval
        res_ = self.res_
        bins = self.bins

        if plot_style is None:
            plot_style = ''
        if windows is None:
            windows = [5, 6, 7, 8, 9]
        windows.sort()
        windows = windows[::-1]
        length = len(windows)

        # compute figure size
        view_bins = view_distance // res_
        view_height = view_bins * 2 ** 0.5 / 2
        fig_height = bins * 2 ** 0.5 / 2

        height = int(abs(self.trans_x_value(0, center_x) - center_x) / 10)
        expand_height = 0
        plot_xticks = False if plot_style == 'line' else True
        if plot_style == 'heatmap':
            if length < 3:
                expand_height = 3 * interval + height * 3
            else:
                expand_height = 3 * interval + height * length
        elif plot_style == 'line' or plot_style == 'directionality':
            expand_height = 3 * interval + height * 4

        if insulation_plot_pos is None:
            figure_height = view_height * 2 + interval
        elif insulation_plot_pos in ['separate', 'sep', 's', 'both', 'b', 'all']:
            figure_height = view_height * 2 + expand_height * 2
        else:
            figure_height = view_height * 2 + expand_height
        figure_width = x_end - x_start + 11 * interval

        k = (figure_height + 6 * interval) / (figure_width + 4 * interval)

        ### create figure ###
        fig_height_add = 4 if plot_style in 'line' else 3 if length < 4 else length

        plt.figure(figsize=(11, 11 * k))

        ### plot HicHeatmap ###
        self.hic_compare_heatmap_plot(view_distance, plot_xticks=plot_xticks)
        fhr = self.figure_height_range

        ### plot insulation ###
        if insulation_plot_pos is not None:
            y0_upper = fhr[1] + 4 * interval
            if plot_style == 'line' or plot_style == 'directionality':
                y0_lower = fhr[0] - 4 * interval - height * 4
            elif plot_style == 'heatmap':
                if length < 3:
                    y0_lower = fhr[0] - 4 * interval - height * 3
                else:
                    y0_lower = fhr[0] - 4 * interval - height * length

            if insulation_plot_pos in ['separate', 'sep', 's', 'both', 'b', 'all']:
                if plot_style in 'directionality':
                    self.hic_directionality_plot(windows, y0=y0_upper, up_down='up', legend=True)
                    self.hic_directionality_plot(windows, y0=y0_lower, up_down='down', legend=False)
                else:
                    self.hic_insulation_plot(windows, plot_style, fig_height_add,
                                             y0=y0_upper, up_down='up', ignore_bins=ignore_bins, legend=True, **kwarg)
                    self.hic_insulation_plot(windows, plot_style, fig_height_add,
                                             y0=y0_lower, up_down='down', ignore_bins=ignore_bins, legend=False, **kwarg)
            elif insulation_plot_pos in 'upper':
                if plot_style in 'directionality':
                    self.hic_directionality_plot(windows, y0=y0_upper, up_down='up', legend=True)
                else:
                    self.hic_insulation_plot(windows, plot_style, fig_height_add,
                                             y0=y0_upper, up_down=insulation_plot_pos,
                                             ignore_bins=ignore_bins, legend=True, **kwarg)
            elif insulation_plot_pos in 'down':
                if plot_style in 'directionality':
                    self.hic_directionality_plot(windows, y0=y0_lower, up_down='down', legend=True)
                else:
                    self.hic_insulation_plot(windows, plot_style, fig_height_add,
                                             y0=y0_lower, up_down=insulation_plot_pos,
                                             ignore_bins=ignore_bins, legend=True, **kwarg)

        ### give lim ###
        fhr = self.figure_height_range
        plt.xlim(x_start - 2 * interval, x_end + 13 * interval)
        plt.ylim(fhr[0] - 4 * interval, fhr[1] + 2 * interval)

    def hic_compare_heatmap_plot(self, view_distance, tickts=None, bar_label_list=None, plot_xticks=False):
        """
        Perform compared matrix's heatmap (rotate 45 degree) plotting.
        :param view_distance: only show interaction below this distance
        :param plot_xticks: control if plot xticks
        :return:
        """
        mat_ = self.mat_
        center_x = self.center_x
        center_y = self.center_y
        interval = self.interval
        x_start = self.x_start
        x_end = self.x_end
        label_bin = self.label_bin
        label_name = self.label_name
        res_ = self.res_
        bins = self.bins

        if tickts is None:
            tickts = self.tickts
        if bar_label_list is None:
            bar_label_list = self.bar_label_list

        view_bins = view_distance // res_
        view_height = view_bins * 2 ** 0.5 / 2
        fig_height = bins * 2 ** 0.5 / 2

        self.figure_height_range = [center_y - view_height, center_y + view_height]

        # plot heatmap
        fh.facile_heatmap(plt, mat_, c='r', cmod=1, rotation=45, canvas=False, islabel=False, normalize='all')
        plt.fill_between([x_start, x_end],
                         [center_y + view_height, center_y + view_height],
                         [center_y + fig_height, center_y + fig_height], color='white')
        plt.fill_between([x_start, x_end],
                         [center_y - view_height, center_y - view_height],
                         [center_y - fig_height, center_y - fig_height], color='white')

        if plot_xticks:
            y_use = center_y - view_height - interval
            plt.plot([x_start, x_end], [y_use, y_use], c='black')
            for i_, name_ in zip(label_bin, label_name):
                i_ = self.trans_x_value(i_, center_x)
                i_ = x_end if i_ > x_end else i_
                plt.plot([i_, i_], [y_use, y_use + 0.03 * center_y], c='black')
                plt.text(i_, y_use - 1.5 * interval, name_, horizontalalignment='center', fontsize=12)

        # hic color bar
        cbar_x0 = x_end + interval * 5
        cb_height = view_height * 2
        cb.colorbar(plt, self.cmap1, cmod=2, height=cb_height, width=cb_height / 10,
                    x0=cbar_x0, y0=center_y - view_height,
                    outer=0.5, ticks=tickts, labels=bar_label_list, cr=False, lc='black')

        cbar_x = cbar_x0 + cb_height / 10
        for i in tickts:
            add = center_y - view_height
            plt.plot([cbar_x, cbar_x + interval / 2],
                     [i * cb_height + add, i * cb_height + add],
                     c='black', lw=1)

        self.cbar_x0 = cbar_x0
        self.cbar_x = cbar_x

    @staticmethod
    def compute_directionality_index(score_A_, score_B_):
        score_E_ = (score_A_ + score_B_) / 2
        part_1 = (score_B_ - score_A_) / abs(score_B_ - score_A_)
        part_2 = (score_A_ - score_E_) ** 2 / score_E_ + (score_B_ - score_E_) ** 2 / score_E_

        return part_1 * part_2

    def get_directionality_index(self, mat_raw_, window_s, up_down='down'):
        """
        :param mat1_raw_: should be unbalanced raw
        :param window_s: window list for directionality analysis, values indicate bins, default is [5, 6, 7, 8, 9, 10]
        :param up_down: plot at 'up' of main heatmap or 'down'
        :return:
        """
        if mat_raw_ is None:
            raise ValueError('directionality index need unbalanced matrix, which should statement while Instantial.')
        mat_raw_len = mat_raw_.shape[0]

        all_directionality_scores = []
        for window in window_s:
            directionality_scores = []
            for i in range(mat_raw_len):
                if i < window or i >= mat_raw_len - window:
                    directionality_scores += [0]
                    continue

                sub_mat = mat_raw_[i - window: i + window + 1, i - window: i + window + 1]
                nans_ = np.isnan(sub_mat)
                sub_mat[nans_] = 0

                if up_down in 'upper':
                    score_A = sub_mat[0, window]
                    score_B = sub_mat[window, -1]
                elif up_down in 'down':
                    score_A = sub_mat[window, 0]
                    score_B = sub_mat[-1, window]

                if score_B == score_A:
                    directionality_score = 0
                else:
                    directionality_score = self.compute_directionality_index(score_A, score_B)
                directionality_scores += [directionality_score]

            all_directionality_scores += [directionality_scores]
        all_directionality_scores = pd.DataFrame(all_directionality_scores, index=window_s)

        return all_directionality_scores

    def hic_directionality_plot(self, windows, y0=None, up_down='down', legend=True):
        """
        plot directionality analysis,
        :param mat_unbalanced: mat_unbalanced
        :param windows: sinlge value or list of window for directionality analysis
        :param y0: y start for directionality plot
        :param up_down: plot at 'up' of main heatmap or 'down'
        :param legend: plot legend or not
        :return:
        """
        center_x = self.center_x
        center_y = self.center_y
        interval = self.interval
        y_use = self.y_use
        x_start = self.x_start
        x_end = self.x_end

        is_int = type(windows) == int
        if is_int:
            windows = [windows]

        directionality_index = self.get_directionality_index(self.mat_unbalanced, window_s=windows, up_down=up_down)
        states, probs, directionality = self.hmm_consensus(directionality_index)

        length = len(windows)
        height = int(abs(self.trans_x_value(0, center_x) - center_x) / 10)

        ## plot directionality fig ##
        cbar_x0 = self.cbar_x0
        cbar_x = self.cbar_x
        expand_height = 3 * interval + height * 4

        if y0 is None:
            y0 = y_use - expand_height
            self.y0 = y0

        c_up = np.array([184, 83, 84]) / 255
        c_down = np.array([69, 117, 70]) / 255

        ## plot hmm heatmap ##
        hmm_hm = np.zeros([3, len(states)])
        for i, val in enumerate(states):
            hmm_hm[val, i] = 1
        hmm_hm = pd.DataFrame(hmm_hm[[2, 0], :])

        fh.facile_heatmap(plt, hmm_hm, c=[c_up, c_down], cmod=[1, 1],
                          islabel=False, canvas=False,  # edgecolor='white', linewidth=0.1,
                          x0=x_start, width=2 ** 0.5, y0=y0, height=height / 2)

        for i, val in enumerate(states):
            if val == 1:
                continue
            prob = probs[i]
            val = 1 if val == 2 else 0
            color = c_up if val == 1 else c_down
            x = self.trans_x_value(i + 0.5, center_x)
            y = y0 + height * (1/4 + val / 2)
            if prob >= 0.9:
                plt.scatter(x, y, s=8, c=['white'], marker='D')

        ytickt_x0 = x_start - interval
        plt.text(ytickt_x0 - interval, y0 + height * 3 / 4, 'HMM',
                 horizontalalignment='center', verticalalignment='center', fontsize=11)
        plt.text(ytickt_x0 - interval, y0 + height * 1 / 4, 'state',
                 horizontalalignment='center', verticalalignment='center', fontsize=12)

        ## plot directionality bar ##
        y_base = y0 + 2.5 * height + interval / 2
        for i, val in enumerate(directionality):
            if val == 0:
                continue
            color = c_up if val > 0 else c_down

            val = val / 50
            val = 1 if val > 1 else val
            val = -1 if val < -1 else val
            val = val * 1.5 * height

            x = self.trans_x_value(i, center_x)
            plt.bar(x, height=val, bottom=y_base, width=2 ** 0.5, color=color)

        ## plot legend ##
        if legend:
            # downstream bias (red)
            fh.facile_heatmap(plt, pd.DataFrame([1]), c=c_up, cmod=1,
                              islabel=False, canvas=False,  # edgecolor='white', linewidth=0.1,
                              x0=x_end + 3 * interval, width=2 ** 0.5, y0=y_base + 0.5 * height, height=height / 2)
            x = x_end + 3 * interval + 2 ** 0.5 / 2
            y = y_base + height * 3/4
            # plt.scatter(x, y, s=8, c=['white'], marker='D')
            plt.text(x + interval, y, 'downstream bias',
                     horizontalalignment='left', verticalalignment='center', fontsize=12)

            # upstream bias (greed)
            fh.facile_heatmap(plt, pd.DataFrame([1]), c=c_down, cmod=1,
                              islabel=False, canvas=False,  # edgecolor='white', linewidth=0.1,
                              x0=x_end + 3 * interval, width=2 ** 0.5, y0=y_base - height, height=height / 2)
            x = x_end + 3 * interval + 2 ** 0.5 / 2
            y = y_base - height * 3/4
            # plt.scatter(x, y, s=8, c=['white'], marker='D')
            plt.text(x + interval, y, 'upstream bias',
                     horizontalalignment='left', verticalalignment='center', fontsize=12)

            # significant
            fh.facile_heatmap(plt, pd.DataFrame([1]), c='gray', cmod=1,
                              islabel=False, canvas=False,  # edgecolor='white', linewidth=0.1,
                              x0=x_end + 3 * interval, width=2 ** 0.5, y0=y0 + height / 4, height=height / 2)
            x = x_end + 3 * interval + 2 ** 0.5 / 2
            y = y0 + height / 2
            plt.scatter(x, y, s=8, c=['white'], marker='D')
            plt.text(x + interval, y, 'significant',
                     horizontalalignment='left', verticalalignment='center', fontsize=12)

        # ytickts
        plt.plot([ytickt_x0, ytickt_x0],
                 [y0 + height + interval / 2, y0 + 4 * height + interval / 2], color='black', lw=1)
        for i, lab in enumerate([-50, 50]):
            y0_use = y0 + interval / 2 + (1 + i * 3) * height
            plt.plot([ytickt_x0 - interval * 0.66, ytickt_x0], [y0_use, y0_use], c='black', lw=1)
            plt.text(ytickt_x0 - interval, y0_use, lab,
                     horizontalalignment='right', verticalalignment='center', fontsize=12)

        plt.text(ytickt_x0 - interval, y_base, 'DI',
                 horizontalalignment='right', verticalalignment='center', fontsize=12)
        # plt.text(self.cbar_x0, y_base, 'DI',
        #          horizontalalignment='left', verticalalignment='center', fontsize=15)


        fhr = self.figure_height_range
        if fhr is not None:
            if up_down in 'upper':
                fhr[1] = fhr[1] + expand_height
            elif up_down in 'down':
                fhr[0] = fhr[0] - expand_height
            self.figure_height_range = fhr

    @staticmethod
    def hmm_infer(directionality):
        """
        Base function to perform hmm for Hi-C directionality index data
        :param directionality: directionality index
        :return:
        """
        up = max(max(directionality) + 1, 12)
        down = min(min(directionality) - 1, -12)
        bins = [down, -10, -5, 5, 10, up]
        obs = pd.cut(directionality, bins, right=False, labels=list(range(len(bins) - 1)),
                     retbins=False, precision=3, include_lowest=True, duplicates='raise')

        HLP = hmm.HMMLearnPred(obs,
                               num_state=3, num_obs=len(bins),
                               epsilon=0.0001, iter_num=1000,
                               seed_id=42)
        HLP.step_M()
        states, probs = HLP.predict()

        states = np.array(states)
        directionality = np.array(directionality)
        state_one = np.argmax(HLP.array_pi)
        rest_state = list(set(range(3)) - set([state_one]))
        corr = np.corrcoef(states[states != state_one], directionality[states != state_one])[0, 1]
        state_zero, state_two = (min(rest_state), max(rest_state)) \
            if corr > 0 \
            else (max(rest_state), min(rest_state))
        trans = {0: state_zero, 1: state_one, 2: state_two}
        states_trans = [trans[i] for i in states]

        return states_trans, probs

    def hmm_consensus(self, directionality_index):
        """
        Perform hmm,
        :param directionality_index: a DataFrame
        :return:
        """
        if directionality_index.shape[0] == 1:
            directionality = list(directionality_index.iloc[0, :])
            states_trans, probs = self.hmm_infer(directionality)

            return states_trans, probs, directionality

        else:
            states_all, probs_all = [], []
            for i_ in range(directionality_index.shape[0]):
                directionality = list(directionality_index.iloc[i_, :])
                states, probs = self.hmm_infer(directionality)

                states_all += [states]
                probs_all += [probs]

            states_all = np.array(states_all)
            probs_all = np.array(probs_all)

            voting = [[sum(probs_all[:, i][states_all[:, i] == j])
                       for j in range(3)]
                      for i in range(states_all.shape[1])]
            voting = np.array(voting)
            consensus_voting = (voting.T / np.sum(voting, axis=1)).T

            consensus_states = np.argmax(consensus_voting, axis=1)
            consensus_probs = np.max(consensus_voting, axis=1)

            norm_probs_all = probs_all / np.sum(probs_all, axis=0)
            consensus_directionality = directionality_index * norm_probs_all
            consensus_directionality = np.sum(consensus_directionality, axis=0)

            return consensus_states, consensus_probs, consensus_directionality

    def get_tad_info(self, chr_name, windows=None):
        if windows is None:
            windows = [10]
        mat_unbalanced = self.mat_unbalanced
        chr_start_pos = self.chr_start_pos
        resolution = self.res_

        directionality_index = self.get_directionality_index(mat_unbalanced, window_s=windows)
        states, probs, directionality = self.hmm_consensus(directionality_index)

        start_open = 0
        end_open = False

        result = []
        prob_segments = []
        prob_segment = []

        for i in range(len(states)):
            if end_open:
                if states[i] != 0:
                    result += [[chr_name] + self.check_tad_range(start_pos, i, np.array(prob_segment),
                                                                 chr_start_pos, resolution, threshold=0.9)]
                    prob_segments += [prob_segment]

                    start_open = False
                    end_open = False
                    prob_segment = []

            if not start_open and states[i] == 2:
                start_open = 2
                start_pos = i
                prob_segment += [probs[i]]
                continue

            if start_open:
                prob_segment += [probs[i]]

            if states[i] == 1:
                start_open -= 1
                start_open = start_open if start_open >= 0 else 0

                if not start_open:
                    prob_segment = []
                continue

            if start_open and states[i] == 0:
                start_open = 2
                end_open = True
                continue

        # result = pd.DataFrame(result, index=[chr_name] * len(result))
        # prob_segments = pd.DataFrame(prob_segments)
        return result

    @staticmethod
    def check_tad_range(sp, ep, prob_segment_, chr_start_pos_, res_, threshold=0.9):
        head_con = True
        tail_con = True

        check_number = min(3, int(len(prob_segment_) / 2))
        check_front = list(prob_segment_[:check_number] >= threshold) + [True]
        check_back = list(prob_segment_[-check_number:] >= threshold)[::-1] + [True]

        start_from = np.argmax(check_front)
        end_from = np.argmax(check_back)
        start_from = start_from if start_from < check_number else -1
        end_from = end_from if end_from < check_number else -1

        if start_from == -1:
            head_con = False
        elif start_from:
            sp += start_from
            prob_segment_ = prob_segment_[start_from:]

        if end_from == -1:
            tail_con = False
        elif end_from:
            ep -= end_from
            prob_segment_ = prob_segment_[:-end_from]

        return [chr_start_pos_ + sp * res_, chr_start_pos_ + ep * res_,
                head_con, tail_con, np.round(np.mean(prob_segment_), 4), chr_start_pos_, sp, ep]


def log_norm_mat(*args):
    """
    A joint log normalize function for single or multiple matrix(s)
    :param args: matrix(s)
    :return: normalized matrix(s), colorbar tickts, colorbarv tickts label
    """
    max_val_list = []
    min_val_list = []
    for mat_ in args:
        mat_ = pd.DataFrame(mat_).fillna(0)

        max_val_list += [np.max(np.max(mat_))]
        min_val_list += [np.min(np.min(mat_[mat_ != 0]))]
    
    max_val_ = max(max_val_list)
    min_val_ = min(min_val_list)
    bottom_ = int(np.ceil(abs(np.log10(min_val_))))
    
    norm_mat_list_ = []
    for mat_ in args:
        mat_ = pd.DataFrame(mat_).fillna(0)
        norm_mat_list_ += [np.log10(mat_ * 10 ** bottom_ + 1)]
        
    # colorbar info
    max_value = np.round(max_val_, 2)
    max_log_value = np.log10(max_value * 10 ** bottom_)
    exp_num = int(max_log_value)
    exp_diff = exp_num - bottom_
    
    # create naive ticks
    bar_label_list_ = ['10^' + str(-i) for i in range(1, bottom_ + 1)][::-1] + [1]
    ticks_ = list(range(bottom_ + 1))

    if exp_diff < 0:
        bar_label_list_ = bar_label_list_[:exp_diff] + [max_value]
        ticks_ = ticks_[:exp_diff] + [max_log_value]
    else:
        bar_label_list_ += ['10^' + str(i) for i in range(1, exp_diff + 1)] + [max_value]
        ticks_ += [i for i in range(bottom_ + 1, bottom_ + 1 + exp_diff)] + [max_log_value]

    # give zero value space
    zero_space = 0.1 ** bottom_ / max_log_value
    ticks_ = np.array(ticks_) / (max_log_value / (1 - zero_space)) + zero_space

    # trim if max_value close to log label
    use_block_length = (len(bar_label_list_) - 2)
    overlap_thres = use_block_length * 0.05
    fold_change_thres = 10 ** overlap_thres

    max_string = str(max_value).replace('.', '')
    thres_string = str(fold_change_thres).replace('.', '')
    min_length = min(len(max_string), len(max_string))
    max_string = int(max_string[:min_length])
    thres_string = int(thres_string[:min_length])
    if max_string < thres_string:
        bar_label_list_ = bar_label_list_[:-2] + bar_label_list_[-1:]
        ticks_ = list(ticks_[:-2]) + list(ticks_[-1:])
    else:
        ticks_ = list(ticks_)
    
    return norm_mat_list_ + [ticks_, bar_label_list_]


def reform_genome_pos(size_, detail=False, remain=1, rounding=2):
    """
    Reform genome distance to more readable
    :param size_: genome distance
    :param detail: more unit
    :param remain: number that remain units
    :param rounding: value that rounded
    :return: reformed genome distance
    """
    if size_ < 0:
        raise ValueError('size can not be a negative number.')
    if size_ < 1000:
        return str(size_) + 'b'
    import math

    magnitude_name = ['b', 'kb', 'mb', 'gb']
    magnitude = int(math.log(size_, 1000))

    if not detail:
        size__ = np.round(size_ / (1000 ** magnitude), rounding)

        return str(size__) + magnitude_name[magnitude]

    else:
        size__ = size_
        infomation = []
        for i_ in range(magnitude):
            size__, size__0 = divmod(size__, 1000)
            infomation += [[size__, size__0]]

        out_num = infomation[-1]
        for info in infomation[:-1][::-1]:
            out_num += [info[1]]

        out_string_ = []
        for num_, name_ in zip(out_num, magnitude_name[: magnitude + 1][::-1]):
            if num_ != 0:
                out_string_ += [str(num_) + name_]

        return ' '.join(out_string_[: remain])


def trans_number(num_):
    string_num = str(num_)
    string_num_out = ''
    i_ = 0
    for i_ in range(0, len(string_num) - 3, 3):
        if i_ == 0:
            string_num_out = ',' + string_num[-3:] + string_num_out
        else:
            string_num_out = ',' + string_num[-(i_ + 3): -i_] + string_num_out

    string_num_out = string_num[:-(i_ + 3)] + string_num_out
    if not string_num_out:
        string_num_out = string_num

    return string_num_out


if __name__ == "__main__":
    import cooler

    ### parameters ###
    resolution = 5000
    chr_name = 'chr12'
    chr_start = '113,172,121'
    chr_end = '113,554,587'
    windows = [5]  # [5, 6, 7, 8, 9, 10]
    balance = True

    # Load the .mcool file
    path1 = 'E:/数据/尚雅芳/Hi_C/coolers-res/huge/S25/priB-HaeIII-allReps-filtered.mcool::/resolutions/'
    path2 = 'E:/数据/尚雅芳/Hi_C/coolers-res/huge/S26/priB-HaeIII-allReps-filtered.mcool::/resolutions/'
    c_obj1 = cooler.Cooler(path1 + str(resolution))
    c_obj2 = cooler.Cooler(path2 + str(resolution))

    # Get the matrix at a certain resolution
    mat1_raw = c_obj1.matrix(balance=True).fetch(chr_name + ':' + chr_start + '-' + chr_end)
    mat2_raw = c_obj2.matrix(balance=True).fetch(chr_name + ':' + chr_start + '-' + chr_end)
    mat1_unbalanced = c_obj1.matrix(balance=False).fetch(chr_name + ':' + chr_start + '-' + chr_end)
    mat2_unbalanced = c_obj2.matrix(balance=False).fetch(chr_name + ':' + chr_start + '-' + chr_end)

    nans = np.isnan(mat1_raw)
    mat1_raw[nans] = 0
    nans = np.isnan(mat2_raw)
    mat2_raw[nans] = 0
    mat1, mat2, tickts, bar_label_list = log_norm_mat(mat1_raw, mat2_raw)

    Hic_ploter = HicHeatmapPlotting(mat1, mat1_raw,
                                    chr_start, chr_end, resolution,
                                    tickts, bar_label_list, mat_unbalanced=mat1_unbalanced)

    # Hic_ploter.plot(windows=windows, plot_style='line')
    Hic_ploter.plot(windows=windows, plot_style='directionality')
    plt.show()
