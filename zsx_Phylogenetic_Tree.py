# -*- coding: UTF-8 -*-
# @Author: Zhang Senxin

import platform
import numpy as np
import pandas as pd
import zsx_ring_plot as rp
import zsx_some_tools as st
import matplotlib.pyplot as plt
from collections import defaultdict


class PhylogeneticTreePlot(object):
    def __init__(self, file=None, silence=False, element=None):
        """
        Plot phylogenetic tree.
        :param file: logging file object
        :param silence: print info or not
        :param element: sequence element, if None, code will auto identify
        """
        system = platform.system()
        if system == 'linux':
            plt.switch_backend('agg')

        self.file = file
        self.silence = silence
        self.element = element

        self.raw_data = None

        self.nodes = []
        self.edges = []
        self.class_dict = {}
        self.counts_dict = {}
        self.node_index_dict = {}
        self.root = ''

        self.distance = {}
        self.layer = []
        self.set_all = set()

        self.height = {}
        self.Father_son = []
        self.Tree = None

        self.is_nps = False

    def plot_pipeline(self, raw_data, save_file, down_threshold=3, up_threshold=1000,
                      color_dic=None, is_nps=False, **kwargs):
        """
        Pipeline to plot phylogenetic tree.
        :param raw_data: standard phylogenetic tree plot data or NPS cluster data, at least have column \'seq\'
        :param save_file: figure save path.
        :param down_threshold: number of unique sequences less than down_threshold will not be drawn
        :param up_threshold:  number of unique sequences more than up_threshold will not be drawn
        :param color_dic: dict contains class and color
        :param is_nps: is raw_data a NPS cluster data
        :param kwargs: plt.savefig arg, dpi is usually used
        :return: raw_data with Read and Node information
        """
        if is_nps:
            raw_data = reform_nps_data(raw_data)
        self.is_nps = is_nps

        run = self.undirected(raw_data, down_threshold=down_threshold, up_threshold=up_threshold)
        if run == 1:
            self.hierarchy()
            self.comput_height()

            # plot
            self.Phylogenetic_Tree(save_file, color_dic=color_dic, **kwargs)
            self.pie_summary(save_file, color_dic=color_dic, **kwargs)

    def undirected(self, raw_data, down_threshold=3, up_threshold=1000, **kwargs):
        """
        Extract the edge and point relationship of the undirected graph.
        :param raw_data: standard phylogenetic tree plot data or NPS cluster data, at least have column \'seq\'
        :param down_threshold: number of unique sequences less than down_threshold will not be drawn
        :param up_threshold:  number of unique sequences more than up_threshold will not be drawn
        :param kwargs: file, silence, and element
        :return: 1, 0, -1; to show if number of unique sequences is fit
        """
        self.raw_data = raw_data

        unique_seq = st.remove_duplicate_list(raw_data.loc[:, 'seq'])
        if len(unique_seq) < down_threshold:
            return 0
        if len(unique_seq) > up_threshold:
            return -1

        element, file, silence = self._fetch_kwargs(['element', 'file', 'silence'], kwargs)

        raw_data = raw_data.reset_index()
        raw_data.index = ['R' + str(i) for i in raw_data.index]

        SDC = st.SeqDistComp(element=element)

        if file is not None:
            st.my_logging(file, str(len(unique_seq)) + ' nodes need to be computed', adding=False)
        if not silence:
            print(len(unique_seq), 'nodes need to be computed', end='\r')

        dis = {}
        set_all = set(list(range(len(unique_seq))))
        set_in = set()
        edges = []
        nodes = ['N' + str(i) for i in range(1, len(unique_seq) + 1)]

        node_index_dict = {}
        index_node_dict = {}
        seq_node_dict = {}
        for cdr, node in zip(unique_seq, nodes):
            da_ = raw_data.loc[raw_data.loc[:, 'seq'] == cdr]
            node_index_dict[node] = [ind for ind in da_.index]
            seq_node_dict[cdr] = node
            for ind in da_.index:
                index_node_dict[ind] = node
        raw_data.loc[:, 'node_id'] = [index_node_dict[ind] for ind in raw_data.index]

        class_dict = {}
        counts_dict = {}
        for ind in raw_data.index:
            class_dict[str(ind)] = raw_data.loc[ind, 'class']
        for ind in raw_data.index:
            counts_dict[str(ind)] = raw_data.loc[ind, 'counts']

        # compute root node: if nps data, root candidate only from IGHM (if it has)
        if self.is_nps and ('IGHM' in list(raw_data.loc[:, 'class'])):
            candidate = list(raw_data.loc[raw_data.loc[:, 'class'] == 'IGHM', 'seq'])
            label = [seq_node_dict[seq] for seq in candidate]
        else:
            candidate = unique_seq
            label = nodes

        seq_all = list(raw_data.loc[:, 'seq'])
        seq_hot, seq_all = SDC.encoding_seqs(candidate, seq_all, encoding='one-hot', flatting=True)

        counts_array = np.array(list(raw_data.loc[:, 'counts']))
        center_seq = np.sum((seq_all.T * counts_array).T / np.sum(counts_array), axis=0)
        center_seq = center_seq[None, :]
        dist = SDC.computing_dist(seq1_coding=seq_hot, seq2_coding=center_seq, method='euclidean')
        dis_center = pd.DataFrame([label, dist[:, 0]]).T
        root = dis_center.sort_values(by=1).iloc[0, 0]

        # compute distance matrix
        dd = SDC.computing_dist(seq1=unique_seq, method='euclidean')
        K = pd.DataFrame(dd)
        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                dis[(nodes[i], nodes[j], i, j)] = K.iloc[i, j]

        # init greedy algorithm
        index = min(dis, key=dis.get)
        edges += [(str(index[0]), str(index[1]), dis[index])]
        set_in.add(index[2])
        set_in.add(index[3])
        set_out = set_all - set_in

        # iterate greedy algorithm
        for i in range(len(unique_seq) - 2):
            distance = []
            for s_in in set_in:
                for s_out in set_out:
                    try:
                        distance += [[dis[nodes[s_in], nodes[s_out], s_in, s_out], s_in, s_out]]
                    except KeyError:
                        distance += [[dis[nodes[s_out], nodes[s_in], s_out, s_in], s_out, s_in]]
            distance = pd.DataFrame(distance)

            mindis = min(distance.iloc[:, 0])
            choose = distance.loc[distance.iloc[:, 0] == min(distance.iloc[:, 0])].iloc[0, :]
            a = nodes[int(choose[1])]
            b = nodes[int(choose[2])]
            edges += [(a, b, mindis)]

            set_in.add(int(choose[1]))
            set_in.add(int(choose[2]))
            set_out = set_all - set_in

        self.nodes = nodes
        self.edges = edges
        self.class_dict = class_dict
        self.counts_dict = counts_dict
        self.node_index_dict = node_index_dict
        self.root = root
        self.raw_data = raw_data

        return 1

    def hierarchy(self):
        """
        Extract hierarchical structure from nodes, edges
        :return: None
        """
        nodes, edges, root = self._fetch_kwargs(['nodes', 'edges', 'root'])
        edges_d = pd.DataFrame(edges)

        distance = {}
        layer = []

        set_all = set(nodes)
        set_in = set()
        distance[root] = 0

        # find sub-root nodes
        connect = pd.concat([edges_d.loc[edges_d.loc[:, 0] == root], edges_d.loc[edges_d.loc[:, 1] == root]])
        connect_nodes = set(list(connect.iloc[:, 0]) + list(connect.iloc[:, 1])) - {root}
        for i in range(connect.shape[0]):
            sss = connect.iloc[i, :2]
            ssr = sss.loc[sss != root].iloc[0]
            distance[ssr] = connect.iloc[i, 2]

        # update
        layer += [{root: list(connect_nodes)}]
        set_in.add(root)
        set_in = set_in | connect_nodes
        set_out = set_all - set_in

        # iterate node levels
        while len(set_out) > 0:
            level_new = {}
            nodes_sum = []
            for node in list(connect_nodes):
                new_connect = pd.concat([edges_d.loc[edges_d.iloc[:, 0] == node],
                                         edges_d.loc[edges_d.iloc[:, 1] == node]])  # all neighbour nodes for one
                new_nodes = set(list(new_connect.iloc[:, 0]) + list(new_connect.iloc[:, 1])) - {node}
                new_nodes = list(new_nodes & set_out)  # # all next layer neighbour nodes for one
                nodes_sum += new_nodes  # update new_nodes
                level_new[node] = new_nodes  # update node dict for this layer

                for new in new_nodes:  # compute distance (x-axis)
                    new_con = pd.concat([new_connect.loc[new_connect.iloc[:, 0] == new],
                                         new_connect.loc[new_connect.iloc[:, 1] == new]])
                    distance[new] = new_con.iloc[0, 2] + distance[node]

            layer += [level_new]
            set_in = set_in | set(nodes_sum)
            set_out = set_all - set_in

            connect_nodes = nodes_sum

        self.distance = distance
        self.layer = layer
        self.set_all = set_all

    def comput_height(self):
        """
        Calculate the ordinate
        :return: None
        """
        layer, set_all = self._fetch_kwargs(['layer', 'set_all'])

        long = 0  # get height (y-axis)
        extremity = []  # save end nodes
        for level, level_dict in enumerate(layer):
            for key, value in level_dict.items():
                if not value:
                    long += 1
                    extremity += [[level, key]]

                elif level == len(layer) - 1:
                    long += len(value)
                    for node in value:
                        extremity += [[level + 1, node]]

        extremity = pd.DataFrame(extremity)
        Tree = pd.DataFrame(np.zeros([len(layer) + 1, long]))

        # get father son relation
        Father_son = []
        for level_dict in layer:
            for key, value in level_dict.items():
                for node in value:
                    if value:
                        Father_son += [[key, node]]
        Father_son = pd.DataFrame(Father_son)

        # backtracking from the last layer of the entire tree structure
        start = 0
        for i in list(range(len(layer) + 1))[::-1][:-1]:
            da = list(extremity.loc[extremity.iloc[:, 0] == i, 1])  # write last layer
            Tree.iloc[i, start:start + len(da)] = da

            father = []
            for node in Tree.iloc[i, :start + len(da)]:  # write upper layer
                father += [Father_son.loc[Father_son.iloc[:, 1] == node].iloc[0, 0]]
            Tree.iloc[i - 1, :start + len(da)] = father

            start += len(da)  # update position

        # sort tree
        Tree_s = Tree.T.copy()
        Tree_s[Tree_s == 0] = '-1'
        Tree_s = Tree_s.sort_values(by=list(range(1, len(layer) + 1)))
        Tree = Tree_s.T
        Tree.columns = list(range(long))

        # compute height (y-axis)
        height = {}

        for node in set_all:
            pos = Tree.index[np.sum(Tree == node, axis=1) != 0][0]
            h = np.mean(Tree.loc[:, Tree.iloc[pos, :] == node].columns)
            height[node] = h

        self.height = height
        self.Father_son = Father_son
        self.Tree = Tree

    def Phylogenetic_Tree(self, save_file, color_dic=None, **kwargs):
        """
        Use such value to plot Phylogenetic Tree.
        :param save_file: figure save file
        :param color_dic: dict contains class and color
        :param kwargs: plt.savefig arg, dpi is usually used
        :return:
        """
        def color():
            return 'lightgrey'

        class_dict, counts_dict, set_all = self._fetch_kwargs(['class_dict', 'counts_dict', 'set_all'])
        distance, height, Father_son = self._fetch_kwargs(['distance', 'height', 'Father_son'])
        node_index_dict, file, silence = self._fetch_kwargs(['node_index_dict', 'file', 'silence'])

        if color_dic is None:
            color_dic = defaultdict(color, {'IGHA1': '#75BDE0', 'IGHA2': '#75BDE0', 'IGHD': 'violet', 'IGHE': 'gold',
                                            'IGHEP1': 'yellow', 'IGHG1': '#B9CC95', 'IGHG2': '#B9CC95',
                                            'IGHG2B': '#B9CC95', 'IGHG2C': '#B9CC95', 'IGHG3': '#B9CC95',
                                            'IGHG4': '#B9CC95', 'IGHGP': 'c', 'IGHM': '#F89B9B', 'Na': 'lightgrey'})

        x_max = max(distance.values())
        y_max = max(height.values())
        xx = x_max + 2
        yy = y_max + 2
        r = yy / xx
        try:
            plt.figure(figsize=(16, 16 * r))
        except ValueError:
            plt.figure(figsize=(16 / r, 16))

        plt.xlim(-1, x_max + 1)
        plt.ylim(-1, y_max + 1)

        for i in range(Father_son.shape[0]):  # draw horizontal lines
            fa = Father_son.iloc[i, 0]
            so = Father_son.iloc[i, 1]
            plt.plot([distance[fa], distance[fa] + 0.5], [height[fa], height[fa]], c='black', lw=1)
            plt.plot([distance[fa] + 0.5, distance[so]], [height[so], height[so]], c='black', lw=1)

        for father_one in Father_son.iloc[:, 0]:  # draw vertical lines.
            fa_data = pd.DataFrame(Father_son.loc[Father_son.iloc[:, 0] == father_one])
            high = [height[fa_data.iloc[i, 1]] for i in range(fa_data.shape[0])]
            sup = max(high)
            inf = min(high)
            plt.plot([distance[father_one] + 0.5, distance[father_one] + 0.5], [inf, sup], c='black', lw=1)

        for node in set_all:  # draw nodes
            index = node_index_dict[node]
            if len(index) == 1:  # single node
                color = color_dic[class_dict[index[0]]]
                r = 0.15 * (np.log10(counts_dict[index[0]]) + 1)
                x = np.arange(-r, +r, 0.001)
                y1 = (r ** 2 - x ** 2) ** 0.5 + height[node]
                y2 = - (r ** 2 - x ** 2) ** 0.5 + height[node]
                x += distance[node]

                plt.fill_between(x, y1, y2, color=color, zorder=10)
                plt.text(distance[node], height[node] + r * 1.05, index[0], fontsize=10, verticalalignment="bottom",
                         horizontalalignment="center", zorder=15)

            else:  # multi node
                x0 = distance[node]
                y0 = height[node]
                x_data = []
                color = []
                for ind in index:
                    color += [color_dic[class_dict[ind]]]
                    x_data += [counts_dict[ind]]
                r = 0.15 * (np.log10(sum(x_data)) + 1)

                rp.ring_plot(x_data,
                             color_list=color,
                             r1=r, r2=0, x0=x0, y0=y0,
                             edge_open=True, lw=1)

                plt.text(distance[node], height[node] + r * 1.05, node, fontsize=10, verticalalignment="bottom",
                         horizontalalignment="center", zorder=15)

        xlim = int(max(distance.values())) + 1
        plt.xticks(np.array(range(0, xlim, 2)), np.array(range(0, xlim, 2)) / 2)

        plt.title(save_file.split('/')[-1].rsplit('.', 1)[0], size=20)
        plt.xlabel('Distance', size=15)
        try:
            plt.savefig(save_file, bbox_inches='tight', transparent=True, **kwargs)
        except MemoryError:
            file, silence = self._fetch_kwargs(['file', 'silence'])
            string = save_file.split('/')[-1] + ' is too huge to save the tree.'
            if file is not None:
                st.my_logging(file, string, adding=True)
            if not silence:
                print(string)

        plt.close()

    def pie_summary(self, save_file, color_dic=None, **kwargs):
        """
        Pie plot for multi-class nodes.
        :param save_file: figure save file
        :param color_dic: dict contains class and color
        :param kwargs: plt.savefig arg, dpi is usually used
        :return:
        """
        def color():
            return 'lightgrey'

        class_dict, counts_dict, set_all, node_index_dict = self._fetch_kwargs(['class_dict', 'counts_dict',
                                                                                'set_all', 'node_index_dict'])

        if color_dic is None:
            color_dic = defaultdict(color, {'IGHA1': '#75BDE0', 'IGHA2': '#75BDE0', 'IGHD': 'violet', 'IGHE': 'gold',
                                            'IGHEP1': 'yellow', 'IGHG1': '#B9CC95', 'IGHG2': '#B9CC95',
                                            'IGHG2B': '#B9CC95', 'IGHG2C': '#B9CC95', 'IGHG3': '#B9CC95',
                                            'IGHG4': '#B9CC95', 'IGHGP': 'c', 'IGHM': '#F89B9B', 'Na': 'lightgrey'})
        node_use = []
        for node in set_all:
            index = node_index_dict[node]
            if len(index) > 1:
                node_use += [[node, index]]

        pies = len(node_use)
        square = int(np.ceil(pies ** 0.5))
        ratio = pies / 160 if pies > 160 else 1
        plt.figure(figsize=(5 * square / ratio, 4 * square / ratio))

        for i, (node, index) in enumerate(node_use):
            ax = plt.subplot(square, square, i + 1)

            x_data = []
            color = []
            for ind in index:
                color += [color_dic[class_dict[ind]]]
                x_data += [counts_dict[ind]]
            r = 0.15 * (np.log10(sum(x_data)) + 1)
            bigger = round(1 / r, 2)

            rp.ring_plot(x_data,
                         color_list=color,
                         r1=1, r2=0,
                         label=index, fs=10, threshold=0.3,
                         show_value=True, fs2=10,
                         edge_open=True, lw=2)

            # bigger info
            plt.text(2.25, 0.85, 'X ' + str(bigger), size=15, verticalalignment="center", horizontalalignment="center")
            # node info
            plt.text(2.25, 1.3, node, fontsize=20, verticalalignment="center", horizontalalignment="center")

            plt.xlim(-2, 3)
            plt.ylim(-2, 2)
            ax.axis('off')

        plt.savefig(save_file.replace('.', '_pie.'), bbox_inches='tight', transparent=True, **kwargs)
        plt.close()

    def _fetch_kwargs(self, arg_name_list, kwargs=None):
        """
        To fetch kwargs, if not given, get from self object
        :param arg_name_list: args names in list
        :param kwargs: kwargs dict, default si None
        :return: arg values in list
        """
        if kwargs is None:
            kwargs = {}

        out_list = []
        for arg in arg_name_list:
            if arg in kwargs.keys():
                value = kwargs[arg]
            else:
                try:
                    value = getattr(self, arg)
                except AttributeError:
                    print('self has no attribute', arg)
                    value = None
            out_list += [value]

        return out_list

    def _check_raw_data(self):
        """
        Check rawdata at least have column \'seq\', and shape in standard format.
        :return: standard data
        """
        if 'seq' not in self.raw_data.columns:
            print('Input data should at least have column \'seq\'.')

        raw_data = self.raw_data
        if 'class' not in raw_data.columns:
            raw_data.loc[:, 'class'] = 'IGHM'
        if 'counts' not in raw_data.columns:
            raw_data.loc[:, 'counts'] = 1


def reform_nps_data(nps_data):
    """
    Trans NPS cluster data to standard phylogenetic tree plot data.
    :param nps_data: NPS cluster data
    :return: standard phylogenetic tree plot data
    """
    raw_data = nps_data.copy()
    raw_data.loc[:, 'seq'] = [raw_data.loc[ab, 'H_CDR3'] + raw_data.loc[ab, 'L_CDR3'] for ab in raw_data.index]
    raw_data.loc[:, 'class'] = raw_data.loc[:, 'Ig_Class']
    raw_data.loc[:, 'counts'] = raw_data.loc[:, 'repeat']

    return raw_data


# demo
if __name__ == '__main__':
    data = st.read_file('E:/数据/张森欣/za/test_data.txt')
    PTP = PhylogeneticTreePlot(element=['A', 'G', 'C', 'T'])
    PTP.plot_pipeline(data, 'E:/数据/张森欣/za/test.png',
                      down_threshold=3, up_threshold=1000, is_nps=False, dpi=400)
    st.write_file('E:/数据/张森欣/za/test_data_new.txt', PTP.raw_data)
