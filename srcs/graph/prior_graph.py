#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/05
# Modified Date: 2024/05/22
#

import os
import sys
import inspect

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import itertools

from tqdm import tqdm  # smart progress meter for loops

import json  # Open standard file format to store data

import networkx as nx  # Software for complex networks (including graphs)
import numpy as np  # Scientific computing
import pandas as pd  # Open-source data analysis and manipulation tool
import matplotlib.pyplot as plt  # Visualisation tool

from srcs.datasets.wrapper import WrapperDatasets

from srcs.utils import check_if_symmetric

from pdb import set_trace as bp  # This is only for debugging

# ----------------------------------------------------------------------------
# Utilities


def correct_filenames_scene(df):
    img_fns = df.iloc[:, 0].tolist()

    l_imgs = []
    for fn in img_fns:
        new_fn = fn.split(".")[0]
        l_imgs.append(new_fn)

    df.iloc[:, 0] = l_imgs

    return df


def get_prior_graph(config, data_wrapper):
    """ """
    try:
        if config["net_params"]["graph_type"] == "obj-only":
            prior_graph = PriorGraphObjectsOnly(config)

        elif config["net_params"]["graph_type"] == "obj_scene":
            prior_graph = PriorGraphSceneObjects(config)

        else:
            raise ValueError("Prior graph type not valid!")
    except (ValueError, IndexError):
        exit("Could not complete request (get_prior_graph).")

    prior_graph.load_adjacency_matrix(
        # data_wrapper.get_data_dir(),
        os.path.join("resources", config["dataset"]),
        "gpa",
        config["params"]["training_mode"],
        config["params"]["fold_id"],
        "ac",
        self_edges=False,
        a_th=config["net_params"]["prior_graph_thresh"],
    )

    return prior_graph


#############################################################################
# Parent class for Prior Knowledge Graphs


class PriorKnowlegeGraphBase(object):
    def __init__(self, config):
        self.root_dir = config["paths"]["root_dir"]

        self.dataset = config["dataset"]

        try:
            with open(
                os.path.join(self.root_dir, "configs", "datasets.json")
            ) as f:
                datasets_config = json.load(f)

            self.data_dir = os.path.join(
                datasets_config["paths"]["data_prefix"],
                datasets_config["datasets"][self.dataset]["data_dir"],
            )
        except ValueError:
            print("Dataset configuration file not correctly loaded!")

        # All categories: -1. Private class: 0. Public class: 1
        # self.category = args.category_mode

        # self.model_name = args.model_name

        # self.partition = args.training_mode
        self.fold_id = config["params"]["fold_id"]
        # self.partitions_fn = args.partitions_fn

        self.n_out_classes = 2  # default value

        # self.get_list_imgs_training_fold()

        # self.b_self_edges = args.self_edges
        self.b_self_edges = True

        self.Gnx = nx.Graph()
        self.graph_d = None
        self.n_edges_added = 0
        self.n_graph_nodes = 2

    def set_nodes_info(self):
        self.n_graph_nodes_2 = self.n_graph_nodes * self.n_graph_nodes
        self.max_n_edges = self.n_graph_nodes * (self.n_graph_nodes - 1) / 2.0
        self.edges_factor = self.n_graph_nodes_2 / 64.0

    def get_dataset_img_labels(self, config, mode):
        """ """
        print("Loading the {:s} dataset ...".format(config["dataset"]))

        training_mode = config["params"]["training_mode"]

        data_wrapper = WrapperDatasets(
            root_dir=config["paths"]["root_dir"],
            data_dir=self.data_dir,
            num_classes=config["net_params"]["num_out_classes"],
            fold_id=config["params"]["fold_id"],
            n_graph_nodes=config["net_params"]["n_graph_nodes"],
            node_feat_size=config["net_params"]["node_feat_size"],
        )

        data_wrapper.load_split_set(
            config["dataset"],
            partition=training_mode,
            mode=mode,
            b_filter_imgs=config["b_filter_imgs"],
        )

        data_split = data_wrapper.get_data_split(mode)

        self.l_imgs = data_split.imgs
        self.l_labels = data_split.labels

        self.n_imgs = len(self.l_imgs)

        if config["dataset"] == "IPD":
            self.data_prefix = data_split.data_prefix

    def reset_n_self_edges(self):
        self.n_edges_added = 0

    def is_graph_sparse(self):
        if self.n_edges_added < self.edges_factor:
            print("Graph is sparse")
            return True

        elif (self.n_edges_added >= self.edges_factor) and (
            self.n_edges_added < self.max_n_edges
        ):
            print("Graph is almost sparse")
            return True

        else:
            print("Graph is dense")
            return False

    # def get_graph_fn(self, suffix=""):
    #     """ """

    #     graph_fn = "prior_graph_f{:d}_c{:d}{:s}.json".format(
    #         self.fold_id, self.n_out_classes, suffix
    #     )

    #     fullpath = os.path.join(
    #         "{:s}".format(self.root_dir),
    #         "assets",
    #         "adjacency_matrix",
    #         self.model_name,
    #         graph_fn,
    #     )

    #     return fullpath

    def load_graph_nx(self, mode="file"):
        """ """
        assert mode in ["adj_mat", "file"]

        if mode == "adj_mat":
            self.Gnx = nx.from_numpy_array(self.adjacency_matrix)
        elif mode == "file":
            fullpath = self.get_graph_fn()
            ## Check if file exists
            self.compute_graph_nx(fullpath)

    def get_n_edges_th(self, a_th):
        """ """
        adj_mat = (self.adjacency_matrix > a_th).astype(int)

        adj_mat[:2, :] = (self.adjacency_matrix[:2, :] > 0).astype(int)
        adj_mat[:, :2] = (self.adjacency_matrix[:, :2] > 0).astype(int)

        G = nx.from_numpy_array(adj_mat)

        return G.number_of_edges(), nx.number_of_isolates(G)

    def compute_graph_nx(self, json_file_path):
        adj_l = json.load(open(json_file_path, "r"))

        edges_l = []

        for k, neighbours in adj_l.items():
            for v in neighbours:
                edges_l.append((k, str(v)))

        self.Gnx.add_edges_from(edges_l)

    def save_graph_as_json(self, suffix=""):
        """
        The function saves the input graph G into a file as JSON format.

        The input graph G is a Python dictionary with the list of nodes as keys
        and a list of node ids as value. This list represents the edges between
        the (key) node and the (value) nodes. The function assumes that the graph
        G is undirected, that is the corresponding adjacency matrix is symmetric
        and for each pair of nodes, the reverse order of the node ids is also
        an edge.

        JSON format is convenient when the graph is undirected, edges do not have
        weights (i.e., either 0 or 1), and their are sparse (i.e., the number of
        edges is << N(N-1)/2, where N is the number of nodes).
        """
        # if n_edges_added < 0.8 * N_EDGES:
        if self.is_graph_sparse():
            out_fn = self.get_graph_fn(suffix)
            print(out_fn)

            dirname = os.path.dirname(out_fn)
            print(dirname)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            G = self.graph_d

            # Convert keys of G from int64 to int
            G = {int(k): [int(i) for i in v] for k, v in G.items()}

            with open(out_fn, "w") as json_file:
                json.dump(G, json_file, indent=4)

        print("Sparse graph saved to JSON file!")

    def print_graph_stats(self):
        # print("Number of graph nodes: {:d}".format(self.Gnx.number_of_nodes()))
        # print("Number of graph edges: {:d}".format(self.Gnx.number_of_edges()))

        print(
            "Total number of possible edges (symmetric, not self-loops): {:d}".format(
                int(self.max_n_edges)
            )
        )
        print(
            "Total number of possible edges: {:d}".format(self.n_graph_nodes_2)
        )
        print("Real number of nodes: {:d}".format(self.n_graph_nodes))

        print("Number of edges added: {:d}".format(self.n_edges_added))

        self.is_graph_sparse()

    ##########################################################################
    #### Adjacency matrix

    def print_adj_mat(self):
        print(self.adjacency_matrix)

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    def load_adjacency_matrix_csv(self, filename):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into NPZ format (numpy file)
        """
        print("Loading adjacency matrix (CSV) ...")

        df = pd.read_csv(filename, sep=",")

        ext_adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        num_edges = df.shape[0]
        for j_iter in range(num_edges):
            idx1 = int(df.loc[j_iter, :][0]) + self.n_out_classes
            idx2 = int(df.loc[j_iter, :][1]) + self.n_out_classes
            value = df.loc[j_iter, :][2]

            ext_adj_mat[idx1, idx2] = value
            ext_adj_mat[idx2, idx1] = value

        self.adjacency_matrix = ext_adj_mat.astype(float)

    def load_adjacency_matrix_json(self, json_file_path):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into JSON format
        """
        print("Loading adjacency matrix (JSON) ...")

        adj_l = json.load(open(json_file_path, "r"))

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        for k, neighbours in adj_l.items():
            for v in neighbours:
                adj_mat[int(k), v] = 1.0
                adj_mat[v, int(k)] = 1.0

        return adj_mat.astype(float)

    def load_weighted_graph(self, filename):
        """-"""
        df = pd.read_csv(filename, sep=",")

        n_nodes = max(df["Node-1"].max(), df["Node-2"].max()) + 1  # O-index
        assert self.n_graph_nodes >= n_nodes

        edge_weights = df.to_numpy()

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        n_rows, n_cols = edge_weights.shape

        for n in range(n_rows):
            idx1 = edge_weights[n][0]
            idx2 = edge_weights[n][1]
            w = edge_weights[n][2]
            adj_mat[int(idx1), int(idx2)] = w

        return adj_mat

    def load_adjacency_matrix(
        self,
        data_dir,
        model_name,
        partition,
        fold_id,
        mode="square_sym",
        self_edges=False,
        a_th=0,
    ):
        """

        Arguments:
            - filename: name of the file where the adjacency matrix is stored and to load.
            - mode:
            - self_edges: boolean to include also self-edges in the graph.

        mode:
            - square_sym: a square and symmetric matrix representing an undirected graph
        """
        print()  # Empty line on the command line

        assert mode in ["square_sym", "ac"]

        adj_mat_fullpath = os.path.join(
            data_dir,
            "graph_data",
            # self.dirname,
            "adj_mat",
        )

        if partition == "crossval":
            adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

        if partition == "final":
            adj_mat_fn = "prior_graph_final"

        if partition == "original":
            adj_mat_fn = "prior_graph_original"

        adjacency_filename = os.path.join(adj_mat_fullpath, adj_mat_fn)

        if mode == "ac":
            adjacency_filename += "_ac.csv"
            print(adjacency_filename)

            self.adjacency_matrix = self.load_weighted_graph(
                adjacency_filename
            )

            # self.adjacency_matrix[self.adjacency_matrix <= a_th] = 0

            # Threshold applied only to the co-occurrence sub-graph
            adj_mat_obj_occ = self.adjacency_matrix.copy()
            adj_mat_obj_occ[
                : self.n_out_classes, :
            ] = 0.0  # set first rows to zero
            adj_mat_obj_occ[
                :, : self.n_out_classes
            ] = 0.0  # set first columns to zero

            adj_mat_obj_occ[adj_mat_obj_occ <= a_th] = 0

            if not check_if_symmetric(adj_mat_obj_occ):
                print("Matrix of co-occurent objects is not symmetric!")

            tmp_idx = self.n_out_classes
            self.adjacency_matrix[tmp_idx:, tmp_idx:] = adj_mat_obj_occ[
                tmp_idx:, tmp_idx:
            ]

            if self_edges == False:
                np.fill_diagonal(self.adjacency_matrix, 0)

        else:
            adjacency_filename += ".json"

            print(adjacency_filename)

            # ext = filename.split(".")[1]
            # assert ext in ["csv", "json"]
            # assert ext == "json"

            adj_mat = self.load_adjacency_matrix_json(adjacency_filename)

            # This makes sure that there are no self-edges (diagonal is 0s)
            if self_edges == False:
                np.fill_diagonal(adj_mat, 0)

            self.adjacency_matrix = adj_mat

        print("Adjacency matrix loaded!")

    def split_adjacency_matrix(self):
        """ """
        adj_mat_obj_occ = self.adjacency_matrix.copy()
        adj_mat_obj_occ[: self.n_out_classes :, :] = 0.0
        adj_mat_obj_occ[:, : self.n_out_classes :] = 0.0

        adj_mat_bipartite = self.adjacency_matrix.copy()
        adj_mat_bipartite[self.n_out_classes :, self.n_out_classes :] = 0.0

        # self.adj_mat_obj_occ = adj_mat_obj_occ.copy()
        # self.adj_mat_bipartite = adj_mat_bipartite.copy()
        return adj_mat_obj_occ, adj_mat_bipartite

    def run_compute_graph(self):
        """ """
        self.get_dataset_img_labels(self.config, "train")

        if self.b_weighted:
            self.compute_weighted_graph_from_file()
            self.save_weighted_prior_graph()
        else:
            self.get_graph_edges_from_files()
            self.save_graph_as_json()

        self.print_graph_stats()


#############################################################################
class PriorGraphObjectsOnly(PriorKnowlegeGraphBase):
    def __init__(self, config):
        super().__init__(config)

        self.dirname = "obj-only"

        self.config = config  # store config for later uses

        self.num_obj_cat = config["net_params"]["num_obj_cat"]

        if config["net_params"].get("num_scene_cat") != None:
            assert config["net_params"]["num_scene_cat"] == 0

        self.num_scene_cat = 0

        self.b_self_edges = config["net_params"]["self_loop"]

        self.b_special_nodes = config["net_params"]["use_class_nodes"]

        if self.b_special_nodes:
            self.n_class_nodes = config["net_params"]["num_out_classes"]
            self.n_graph_nodes = (
                self.num_obj_cat + self.num_scene_cat + self.n_class_nodes
            )

            self.b_directed = True  # Only for special nodes

            self.n_out_classes = self.n_class_nodes
        else:
            self.n_graph_nodes = self.num_obj_cat + self.num_scene_cat

            self.b_directed = False

        self.b_weighted = True

        self.category = config["category_mode"]

        self.set_nodes_info()

    def add_edge_weight(self, idx1, idx2, prior_graph, weights, symm=False):
        """ """
        if idx1 in prior_graph:
            if idx2 not in prior_graph[idx1]:
                prior_graph[int(idx1)].append(idx2)
                weights[int(idx1)][int(idx2)] = 1

                if symm:
                    self.n_edges_added += 1
            else:
                weights[int(idx1)][int(idx2)] += 1
        else:
            prior_graph[int(idx1)] = [idx2]
            weights[int(idx1)][int(idx2)] = 1

            if symm:
                self.n_edges_added += 1

        return prior_graph, weights

    def normalise_prior_graph_weights(self, weights, cats_occ):
        """ """
        for n in range(self.n_graph_nodes):
            for key, val in weights[n].items():
                if self.b_special_nodes and key in range(self.n_out_classes):
                    weights[n][key] = val / cats_occ[key]
                    # weights[n][key] = val / cats_occ[n]
                else:
                    weights[n][key] = cats_occ[n] and val / cats_occ[n] or 0

        return weights

    def compute_weighted_graph_from_file(self):
        """
        Compute the weighted graph of objects and special nodes from the training set.

        For each image in the training split of a given dataset, the co-occurence
        of object categories identified in the
        """

        prior_graph = dict()
        weights = dict()

        for n in range(self.n_graph_nodes):
            prior_graph[n] = []
            weights[n] = dict()

        # Store the number of occurences of each category across the whole
        # training set
        cats_occ = np.zeros(self.n_graph_nodes)

        for img_idx in tqdm(range(self.n_imgs)):
            if self.dataset == "IPD":
                fullpath = os.path.join(
                    self.data_prefix[img_idx],
                    "graph_data",
                    self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )
            else:
                fullpath = os.path.join(
                    self.data_dir,
                    "graph_data",
                    self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )

            retval = False

            node_feats = json.load(open(fullpath))
            node_cat = []

            for node in node_feats:
                n_id = node["node_id"]

                # This is to add to the list only the concept nodes (objects and scenes)
                # whose cardinality is greater than 0
                # (or simply to exclude the privacy nodes in the format of the node features)
                if node["node_feature"][0] > 0:
                    node_cat.append(n_id)

                    if not retval:
                        retval = True

            # Continue if the image does not contain any concept
            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(node_cat)
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            # Count the occurrence of the number of categories and
            # add self-edges if the corresponding Boolean parameter was set
            for idx in img_cats_unique:
                cats_occ[idx] += 1

                if self.b_self_edges:
                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                prior_graph, weights = self.add_edge_weight(
                    edge[0], edge[1], prior_graph, weights
                )

                # Other direction (symmetric)
                prior_graph, weights = self.add_edge_weight(
                    edge[1], edge[0], prior_graph, weights, True
                )

            if self.b_special_nodes:
                for idx in img_cats_unique:
                    for sp_node_id in range(self.n_out_classes):
                        # Link image nodes to privacy class based on label of the image
                        if sp_node_id == self.l_labels[img_idx]:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights
                            )

                            if not self.b_directed:
                                # prior_graph, weights = self.add_edge_weight(
                                #     idx, sp_node_id, prior_graph, weights, True
                                # )
                                # New corrected version
                                prior_graph, weights = self.add_edge_weight(
                                    sp_node_id, idx, prior_graph, weights, True
                                )

        if self.b_special_nodes:
            for sp_node_id in range(self.n_out_classes):
                cats_occ[sp_node_id] = (
                    np.array(self.l_labels) == sp_node_id
                ).sum()

        weights = self.normalise_prior_graph_weights(weights, cats_occ)

        # self.graph_d = dict(sorted(prior_graph.items()))
        self.graph_d = dict(sorted(weights.items()))

    def save_weighted_prior_graph(self):
        """
        Save the weighted prior graph to file in CSV format.

        The weighted prior graph is automatically saved in the pre-defined
        directory ``/workdir/assets/adjacency_matrix/<model_name>/``, where
        <model_name> is the name of the model (e.g., gpa). The file is saved
        under the name prior_graph_weighted_fX_cY.csv, where X is the fold ID
        and Y is the number of classes, when the prior graph is computed from
        all the images in the training set. Otherwise, the file is saved under
        the name name prior_graph_weighted_fX_cY_clsZ.csv, where Z is the
        output class number (e.g., 0 for private and 1 for public).

        The file is saved as a list of edges with their corresponding weights
        for each row. The columns of the CSV file are:
        - Node-1: the ID of the source node (float).
        - Node-2: the ID of the target node (float).
        - Weight: weight of the edge (float).
        """

        if self.category == -1:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}.csv".format(
                self.fold_id, self.n_out_classes
            )
        else:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}_cls{:d}.csv".format(
                self.fold_id, self.n_out_classes, self.category
            )

        fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            self.dirname,
            "adj_mat",
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        edge_weight = np.array([])
        for n1, val in self.graph_d.items():
            for n2, w in sorted(val.items()):
                row = np.array([n1, n2, w])

                if edge_weight.size == 0:
                    edge_weight = np.hstack((edge_weight, row))
                else:
                    edge_weight = np.vstack((edge_weight, row))

        headers = ["Node-1", "Node-2", "Weight"]
        pd.DataFrame(edge_weight).to_csv(fullpath, header=headers, index=None)


#############################################################################


class PriorGraphSceneObjects(PriorKnowlegeGraphBase):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.dirname = "obj_scene"

        self.num_obj_cat = config["net_params"]["num_obj_cat"]
        self.num_scene_cat = config["net_params"]["num_scene_cat"]

        self.b_self_edges = config["net_params"]["self_loop"]

        self.b_special_nodes = config["net_params"]["use_class_nodes"]

        if self.b_special_nodes:
            self.n_class_nodes = config["net_params"]["num_out_classes"]

            self.n_graph_nodes = (
                self.num_obj_cat + self.num_scene_cat + self.n_class_nodes
            )

            self.b_directed = True  # Only for special nodes

            self.n_out_classes = self.n_class_nodes
        else:
            self.n_graph_nodes = self.num_obj_cat + self.num_scene_cat

            self.b_directed = False

        self.b_weighted = True

        self.category = config["category_mode"]

        self.set_nodes_info()

    def add_edge_weight(self, idx1, idx2, prior_graph, weights, symm=False):
        """ """
        if idx1 in prior_graph:
            if idx2 not in prior_graph[idx1]:
                prior_graph[int(idx1)].append(idx2)
                weights[int(idx1)][int(idx2)] = 1

                if symm:
                    self.n_edges_added += 1
            else:
                weights[int(idx1)][int(idx2)] += 1
        else:
            prior_graph[int(idx1)] = [idx2]
            weights[int(idx1)][int(idx2)] = 1

            if symm:
                self.n_edges_added += 1

        return prior_graph, weights

    def normalise_prior_graph_weights(self, weights, cats_occ):
        """ """
        for n in range(self.n_graph_nodes):
            for key, val in weights[n].items():
                if self.b_special_nodes and key in range(self.n_out_classes):
                    weights[n][key] = val / cats_occ[key]
                    # weights[n][key] = val / cats_occ[n]
                else:
                    weights[n][key] = cats_occ[n] and val / cats_occ[n] or 0

        return weights

    def compute_weighted_graph_from_file(self):
        """
        Compute the weighted graph of objects and special nodes from the training set.

        For each image in the training split of a given dataset, the co-occurence
        of object categories identified in the
        """

        prior_graph = dict()
        weights = dict()

        for n in range(self.n_graph_nodes):
            prior_graph[n] = []
            weights[n] = dict()

        # Store the number of occurences of each category across the whole
        # training set
        cats_occ = np.zeros(self.n_graph_nodes)

        for img_idx in tqdm(range(self.n_imgs)):
            if self.dataset == "IPD":
                fullpath = os.path.join(
                    "resources",
                    # self.dataset,
                    self.data_prefix[img_idx],
                    "graph_data",
                    # self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )
            else:
                fullpath = os.path.join(
                    "resources",
                    self.dataset,
                    "graph_data",
                    # self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )

            retval = False

            node_feats = json.load(open(fullpath))
            node_cat = []

            for node in node_feats:
                n_id = node["node_id"]

                # This is to add to the list only the concept nodes (objects and scenes)
                # whose cardinality is greater than 0
                # (or simply to exclude the privacy nodes in the format of the node features)
                if node["node_feature"][0] > 0:
                    node_cat.append(n_id)

                    if not retval:
                        retval = True

            # Continue if the image does not contain any concept
            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(node_cat)
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            # Count the occurrence of the number of categories and
            # add self-edges if the corresponding Boolean parameter was set
            for idx in img_cats_unique:
                cats_occ[idx] += 1

                if self.b_self_edges:
                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                prior_graph, weights = self.add_edge_weight(
                    edge[0], edge[1], prior_graph, weights
                )

                # Other direction (symmetric)
                prior_graph, weights = self.add_edge_weight(
                    edge[1], edge[0], prior_graph, weights, True
                )

            if self.b_special_nodes:
                for idx in img_cats_unique:
                    for sp_node_id in range(self.n_out_classes):
                        # Link image nodes to privacy class based on label of the image
                        if sp_node_id == self.l_labels[img_idx]:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights
                            )

                            if not self.b_directed:
                                # prior_graph, weights = self.add_edge_weight(
                                #     idx, sp_node_id, prior_graph, weights, True
                                # )
                                # New corrected version
                                prior_graph, weights = self.add_edge_weight(
                                    sp_node_id, idx, prior_graph, weights, True
                                )

        if self.b_special_nodes:
            for sp_node_id in range(self.n_out_classes):
                cats_occ[sp_node_id] = (
                    np.array(self.l_labels) == sp_node_id
                ).sum()

        weights = self.normalise_prior_graph_weights(weights, cats_occ)

        g_nx = nx.DiGraph()
        for idx1 in range(self.n_graph_nodes):
            for key, val in weights[idx1].items():
                g_nx.add_edge(idx1, key, weight=val)

        self.adjacency_matrix = nx.to_numpy_array(g_nx)
        print(check_if_symmetric(self.adjacency_matrix))

        self.graph_d = dict(sorted(weights.items()))

    def save_weighted_prior_graph(self):
        """
        Save the weighted prior graph to file in CSV format.

        The weighted prior graph is automatically saved in the pre-defined
        directory ``/workdir/assets/adjacency_matrix/<model_name>/``, where
        <model_name> is the name of the model (e.g., gpa). The file is saved
        under the name prior_graph_weighted_fX_cY.csv, where X is the fold ID
        and Y is the number of classes, when the prior graph is computed from
        all the images in the training set. Otherwise, the file is saved under
        the name name prior_graph_weighted_fX_cY_clsZ.csv, where Z is the
        output class number (e.g., 0 for private and 1 for public).

        The file is saved as a list of edges with their corresponding weights
        for each row. The columns of the CSV file are:
        - Node-1: the ID of the source node (float).
        - Node-2: the ID of the target node (float).
        - Weight: weight of the edge (float).
        """

        if self.category == -1:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}.csv".format(
                self.fold_id, self.n_out_classes
            )
        else:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}_cls{:d}.csv".format(
                self.fold_id, self.n_out_classes, self.category
            )

        fullpath = os.path.join(
            "resources",
            self.dataset,
            "graph_data",
            # self.dirname,
            "adj_mat",
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        edge_weight = np.array([])
        for n1, val in self.graph_d.items():
            for n2, w in sorted(val.items()):
                row = np.array([n1, n2, w])

                if edge_weight.size == 0:
                    edge_weight = np.hstack((edge_weight, row))
                else:
                    edge_weight = np.vstack((edge_weight, row))

        headers = ["Node-1", "Node-2", "Weight"]
        pd.DataFrame(edge_weight).to_csv(fullpath, header=headers, index=None)


##############################################################################
class PriorKnowlegeGraphGPA(PriorKnowlegeGraphBase):
    def __init__(self, config):
        super().__init__(config)

        # print(self.model_name)
        # print("Graph Privacy Advisor (GPA)")
        self.dirname = ""

        self.num_obj_cat = config["net_params"]["num_obj_cat"]
        self.n_out_classes = config["net_params"]["num_out_classes"]

        self.n_graph_nodes = self.num_obj_cat + self.n_out_classes

        self.config = config

        self.b_special_nodes = config["net_params"]["use_class_nodes"]
        if self.b_special_nodes:
            self.n_class_nodes = config["net_params"]["num_out_classes"]
            self.n_graph_nodes = self.num_obj_cat + self.n_class_nodes

            self.b_directed = True  # Only for special nodes

            self.n_out_classes = self.n_class_nodes
        else:
            self.n_graph_nodes = self.num_obj_cat

            self.b_directed = False

        self.category = config["category_mode"]

        self.b_weighted = True

        self.set_nodes_info()

    def get_image_objects(self, fullpath):
        """ """
        try:
            objs = json.load(open(fullpath))
        except:
            # print("Missing object image: {:s}".format(fullpath))
            # missing_object_img.append(img_name)
            return False, None

        if len(objs["categories"]) == 0:
            # print("Image with no detected objects: {:s}".format(fullpath))
            # missing_object_img.append(img_name)
            return False, None

        return True, objs

    def add_edge_weight(self, idx1, idx2, prior_graph, weights, symm=False):
        """ """
        if idx1 in prior_graph:
            if idx2 not in prior_graph[idx1]:
                prior_graph[int(idx1)].append(idx2)
                weights[int(idx1)][int(idx2)] = 1

                if symm:
                    self.n_edges_added += 1
            else:
                weights[int(idx1)][int(idx2)] += 1
        else:
            prior_graph[int(idx1)] = [idx2]
            weights[int(idx1)][int(idx2)] = 1

            if symm:
                self.n_edges_added += 1

        return prior_graph, weights

    def normalise_prior_graph_weights(self, weights, cats_occ):
        """ """
        for n in range(self.n_graph_nodes):
            for key, val in weights[n].items():
                if self.b_special_nodes and key in range(self.n_out_classes):
                    weights[n][key] = val / cats_occ[key]
                    # weights[n][key] = val / cats_occ[n]
                else:
                    weights[n][key] = val / cats_occ[n]

        return weights

    def compute_weighted_graph_from_file(self):
        """
        Compute the weighted graph of objects and special nodes from the training set.

        For each image in the training split of a given dataset, the co-occurence
        of object categories identified in the
        """

        prior_graph = dict()
        weights = dict()

        for n in range(self.n_graph_nodes):
            prior_graph[n] = []
            weights[n] = dict()

        # Store the number of occurences of each category across the whole
        # training set
        cats_occ = np.zeros(self.n_graph_nodes)

        for img_idx in tqdm(range(self.n_imgs)):
            if self.dataset == "IPD":
                fullpath = os.path.join(
                    self.data_prefix[img_idx],
                    "dets",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )
            else:
                fullpath = os.path.join(
                    self.data_dir,
                    "dets",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )

            retval, objs = self.get_image_objects(fullpath)

            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(objs["categories"]) + self.n_out_classes
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            if self.b_self_edges:
                for idx in img_cats_unique:
                    cats_occ[idx] += 1

                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                prior_graph, weights = self.add_edge_weight(
                    edge[0], edge[1], prior_graph, weights
                )

                # Other direction (symmetric)
                prior_graph, weights = self.add_edge_weight(
                    edge[1], edge[0], prior_graph, weights, True
                )

            if self.b_special_nodes:
                for idx in img_cats_unique:
                    for sp_node_id in range(self.n_out_classes):
                        if sp_node_id == self.l_labels[img_idx]:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights
                            )

                        if not self.b_directed:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights, True
                            )

        if self.b_special_nodes:
            for sp_node_id in range(self.n_out_classes):
                cats_occ[sp_node_id] = (
                    np.array(self.l_labels) == sp_node_id
                ).sum()

        weights = self.normalise_prior_graph_weights(weights, cats_occ)

        # self.graph_d = dict(sorted(prior_graph.items()))
        self.graph_d = dict(sorted(weights.items()))

    def get_graph_edges_from_files(self):
        """
        Compute the edges of the graphs with only COCO object categories as nodes.
        Detected objects are retrieved from the input file.

        The prior graph is computed as an adjacency list, i.e., for each node id,
        we store a list of node id connected to the node under consideration.
        For compacteness of storage and given the symmetric form of the
        undirected graph, only one direction is saved. This means that if exists an
        edge (2,3), we only store the following list 2: [3, ...] and not the viceversa
        3: [2, ...].
        """

        prior_graph = dict()
        for n in range(self.n_graph_nodes):
            prior_graph[n] = []

        # missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            retval, objs = self.get_image_objects(self.l_imgs[img_idx])

            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(objs["categories"]) + self.n_out_classes
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            if self.b_self_edges:
                for idx in img_cats_unique:
                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                if edge[0] in prior_graph:
                    if edge[1] not in prior_graph[edge[0]]:
                        prior_graph[int(edge[0])].append(edge[1])
                        self.n_edges_added += 1
                else:
                    prior_graph[int(edge[0])] = [edge[1]]
                    self.n_edges_added += 1

        self.graph_d = dict(sorted(prior_graph.items()))

        # print(
        #     "Number of missing object images: {:d}".format(
        #         len(missing_object_img)
        #     )
        # )

    def save_weighted_prior_graph(self):
        """
        Save the weighted prior graph to file in CSV format.

        The weighted prior graph is automatically saved in the pre-defined
        directory ``/workdir/assets/adjacency_matrix/<model_name>/``, where
        <model_name> is the name of the model (e.g., gpa). The file is saved
        under the name prior_graph_weighted_fX_cY.csv, where X is the fold ID
        and Y is the number of classes, when the prior graph is computed from
        all the images in the training set. Otherwise, the file is saved under
        the name name prior_graph_weighted_fX_cY_clsZ.csv, where Z is the
        output class number (e.g., 0 for private and 1 for public).

        The file is saved as a list of edges with their corresponding weights
        for each row. The columns of the CSV file are:
        - Node-1: the ID of the source node (float).
        - Node-2: the ID of the target node (float).
        - Weight: weight of the edge (float).
        """

        if self.category == -1:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}.csv".format(
                self.fold_id, self.n_out_classes
            )
        else:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}_cls{:d}.csv".format(
                self.fold_id, self.n_out_classes, self.category
            )

        fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            "{:d}-class".format(self.n_out_classes),
            "gpa",
            "adj_mat",
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        edge_weight = np.array([])
        for n1, val in self.graph_d.items():
            for n2, w in sorted(val.items()):
                row = np.array([n1, n2, w])

                if edge_weight.size == 0:
                    edge_weight = np.hstack((edge_weight, row))
                else:
                    edge_weight = np.vstack((edge_weight, row))

        headers = ["Node-1", "Node-2", "Weight"]
        pd.DataFrame(edge_weight).to_csv(fullpath, header=headers, index=None)

    def run_compute_graph(self):
        """ """

        if self.b_weighted:
            self.compute_weighted_graph_from_file()
            self.save_weighted_prior_graph()
        else:
            self.get_graph_edges_from_files()
            self.save_graph_as_json()

        self.print_graph_stats()

    def run_graph_analysis(self):
        self.load_graph_nx()
        self.print_graph_stats()
        # self.save_graph_node_degrees()

    def load_adjacency_matrix_csv(self, filename):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into NPZ format (numpy file)
        """
        print("Loading adjacency matrix (CSV) ...")

        df = pd.read_csv(filename, sep=",")

        ext_adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        num_edges = df.shape[0]
        for j_iter in range(num_edges):
            idx1 = int(df.loc[j_iter, :][0]) + self.n_out_classes
            idx2 = int(df.loc[j_iter, :][1]) + self.n_out_classes
            value = df.loc[j_iter, :][2]

            ext_adj_mat[idx1, idx2] = value
            ext_adj_mat[idx2, idx1] = value

        self.adjacency_matrix = ext_adj_mat.astype(float)

    def load_adjacency_matrix_json(self, json_file_path):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into JSON format
        """
        print("Loading adjacency matrix (JSON) ...")

        adj_l = json.load(open(json_file_path, "r"))

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        for k, neighbours in adj_l.items():
            for v in neighbours:
                adj_mat[int(k), v] = 1.0
                adj_mat[v, int(k)] = 1.0

        return adj_mat.astype(float)

    def load_weighted_graph(self, filename):
        """-"""
        df = pd.read_csv(filename, sep=",")

        n_nodes = max(df["Node-1"].max(), df["Node-2"].max()) + 1
        assert self.n_graph_nodes == n_nodes

        edge_weights = df.to_numpy()

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        n_rows, n_cols = edge_weights.shape

        for n in range(n_rows):
            idx1 = edge_weights[n][0]
            idx2 = edge_weights[n][1]
            w = edge_weights[n][2]
            adj_mat[int(idx1), int(idx2)] = w

        return adj_mat

        # def get_adjacency_matrix_filename(self, dataset, model_name):
        """ """
        # assert dataset in ["PicAlert", "VISPR", "PrivacyAlert", "IPD"]

        # if dataset == "PicAlert":
        #     dataset_name = "picalert"
        # elif dataset == "VISPR":
        #     dataset_name = "vispr"
        # elif dataset == "PrivacyAlert":
        #     dataset_name = "privacyalert"
        # elif dataset == "IPD":
        #     dataset_name = "ipd"

        # dataset_name = "ipd" # TO BE REMOVED

    # def load_adjacency_matrix(
    #     self,
    #     data_dir,
    #     model_name,
    #     partition,
    #     fold_id,
    #     mode="square_sym",
    #     self_edges=False,
    # ):
    #     """

    #     Arguments:
    #         - filename: name of the file where the adjacency matrix is stored and to load.
    #         - mode:
    #         - self_edges: boolean to include also self-edges in the graph.

    #     mode:
    #         - square_sym: a square and symmetric matrix representing an undirected graph
    #     """
    #     print()  # Empty line on the command line

    #     assert mode in [
    #         "square_sym",
    #         "ac",
    #     ]

    #     adj_mat_fullpath = os.path.join(
    #         data_dir,
    #         "graph_data",
    #         "{:d}-class".format(self.n_out_classes),
    #         model_name,
    #         "adj_mat",
    #     )

    #     if partition == "crossval":
    #         adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

    #     if partition == "final":
    #         adj_mat_fn = "prior_graph_final"

    #     if partition == "original":
    #         adj_mat_fn = "prior_graph_original"
    #         # adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

    #     adjacency_filename = os.path.join(adj_mat_fullpath, adj_mat_fn)

    #     if mode == "ac":
    #         adjacency_filename += "_ac.csv"
    #         # print(adjacency_filename)

    #         self.adjacency_matrix = self.load_weighted_graph(
    #             adjacency_filename
    #         )

    #         th = 0
    #         self.adjacency_matrix[self.adjacency_matrix <= th] = 0

    #     else:
    #         adjacency_filename += ".json"

    #         # ext = filename.split(".")[1]
    #         # assert ext in ["csv", "json"]
    #         # assert ext == "json"

    #         adj_mat = self.load_adjacency_matrix_json(adjacency_filename)

    #         # This makes sure that there are no self-edges (diagonal is 0s)
    #         if self_edges == False:
    #             np.fill_diagonal(adj_mat, 0)

    #         self.adjacency_matrix = adj_mat

    #     print("Adjacency matrix loaded!")

    def split_adjacency_matrix(self):
        """ """
        adj_mat_obj_occ = self.adjacency_matrix.copy()
        adj_mat_obj_occ[: self.n_out_classes :, :] = 0.0
        adj_mat_obj_occ[:, : self.n_out_classes :] = 0.0

        adj_mat_bipartite = self.adjacency_matrix.copy()
        adj_mat_bipartite[self.n_out_classes :, self.n_out_classes :] = 0.0

        # self.adj_mat_obj_occ = adj_mat_obj_occ.copy()
        # self.adj_mat_bipartite = adj_mat_bipartite.copy()
        return adj_mat_obj_occ, adj_mat_bipartite


#############################################################################


class PriorKnowlegeGraphGIP(PriorKnowlegeGraphBase):
    def __init__(self, args):
        super().__init__(args)

        self.n_nodes = args.n_obj_cats + self.n_privacy_cls
        self.n_nodes_2 = self.n_nodes * self.n_nodes
        self.max_n_edges = self.n_nodes * (self.n_nodes - 1) / 2

        self.n_obj_cats = args.n_obj_cats

    def get_graph_edges_from_files(self):
        """
        Compute the edges of the graphs with only COCO object categories as nodes.
        Detected objects are retrieved from the input file.
        """
        # Private: 0; Public: 1 (according to the manifest)
        if self.n_privacy_cls == 2:
            print("Labels: Private: 0; Public: 1 (according to the manifest)")
            n_imgs_pri = len(self.l_labels[self.l_labels == 0].tolist())
            n_imgs_pub = len(self.l_labels[self.l_labels == 1].tolist())
        elif self.n_privacy_cls == 3:
            print(
                "Labels: Private: 0; Undecidable: 1; Public: 2 (according to the manifest)"
            )
            n_imgs_pri = len(self.l_labels[self.l_labels == 0].tolist())
            n_imgs_und = len(self.l_labels[self.l_labels == 1].tolist())
            n_imgs_pub = len(self.l_labels[self.l_labels == 2].tolist())
        else:
            print("Cannot handle number of classes different from 2 or 3!")
            return

        freq_mat = np.zeros([self.n_privacy_cls, self.n_obj_cats])

        missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            img_name = self.l_imgs[img_idx]
            label = self.l_labels[img_idx]

            fullpath = os.path.join(
                self.root_dir,
                "assets",
                "obj_det",
                img_name.split(".")[0] + ".json",
            )

            try:
                objs = json.load(open(fullpath))
            except:
                # print("Missing object image: {:s}".format(fullpath))
                missing_object_img.append(img_name)
                continue

            if len(objs["categories"]) == 0:
                # print("Image with no detected objects: {:s}".format(fullpath))
                missing_object_img.append(img_name)
                continue

            img_cats = np.array(objs["categories"])

            # Even if the are multiple instances for the same category, the next
            # operation adds only 1
            # bp()
            print(len(img_cats))

            freq_mat[label, img_cats] += 1

        if self.n_privacy_cls == 2:
            freq_mat[0, :] /= n_imgs_pri
            freq_mat[1, :] /= n_imgs_pub
        elif self.n_privacy_cls == 3:
            freq_mat[0, :] /= n_imgs_pri
            freq_mat[1, :] /= n_imgs_und
            freq_mat[2, :] /= n_imgs_pub

        self.n_edges_added = np.count_nonzero(freq_mat)

        self.graph_d = freq_mat

        print(
            "Number of missing object images: {:d}".format(
                len(missing_object_img)
            )
        )

    def save_graph_as_csv(self):
        """
        Save the weighted, undirected, bipartite graph as an adjacency matrix.

        Given the special type of graph and simplicity, we can simply save a
        block of the adjacency matrix (relation between the public/private
        node with the object categories). For the .csv file, a row is an object
        category (80 COCO categories in total) and the two columns are the
        private and public labels of images. Each cell of the matrix provide
        the frequency ([0,1]) of the category with respect to the public/private
        label.
        """
        graph_fn = "prior_graph_f{:d}_c{:d}.csv".format(
            self.fold_id, self.n_privacy_cls
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "assets",
            "adjacency_matrix",
            self.model_name,
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if self.n_privacy_cls == 2:
            headers = ["Private", "Public"]
        elif self.n_privacy_cls == 3:
            headers = ["Private", "Undecidable", "Public"]

        pd.DataFrame(self.graph_d.transpose()).to_csv(
            fullpath, header=headers, index=None
        )

    def run_compute_graph(self):
        self.get_graph_edges_from_files()
        self.print_graph_stats()
        self.save_graph_as_csv()


