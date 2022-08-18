#coding=utf-8
"""
Anonymous author
"""

import numpy as np
import networkx as nx
import sys

import torch
from torch.utils.data import Dataset


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output



class PretrainQm9Dataset(Dataset):
    def __init__(self, node_features, adj_features, dts_features, dts_adj_features, mol_sizes):
        self.n_molecule = node_features.shape[0]  # 249440
        self.node_features = node_features  # 2494400 * 38
        self.adj_features = adj_features  # 249440 * 3 * 38 * 38
        self.mol_sizes = mol_sizes  # 249440
        self.dts_features = dts_features  # 2494400 * 38
        self.dts_adj_features = dts_adj_features  # 249440 * 38 * 38
        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 0]
        self.max_size = self.node_features.shape[1]  # 38
        self.node_dim = len(self.atom_list)  # 10

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy()  # (N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32)  # (3, N, N)
        dts_feature_copy = self.dts_features[idx].copy().astype(np.float32)  # (N)
        dts_adj_features = self.dts_adj_features[idx].copy().astype(np.float32)  # (N)
        mol_size = self.mol_sizes[idx]

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size]  # (mol_size, mol_size)
        local_perm = np.random.permutation(mol_size)  # (first perm graph)生成一个随机原子序列
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]  # 按对应原子序列对 边矩阵重排
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0])  # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx))  # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)])

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        dts_feature_copy = dts_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)

        for i in range(3):
            adj_feature_copy[i] = np.where(adj_feature_copy[i] > 0, dts_adj_features, 0)
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]

        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32)  # (N,10)
        for i in range(self.max_size):
            value_dts = dts_feature_copy[i]
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = value_dts
        node_feature = node_feature[:, :-1]  # (N, 9)

        num_no_edge = adj_feature_copy[adj_feature_copy > 0].mean()

        t_adj = np.sum(adj_feature_copy, axis=0, keepdims=True)
        adj_ones = np.ones_like(t_adj)
        con_adj = np.where(t_adj > 0, adj_ones, 0)
        adj_feature = np.concatenate([adj_feature_copy, 1 - con_adj], axis=0).astype(np.float32)  # (4, N, N)
        adj_feature[3] *= num_no_edge

        for i in range(4):
            adj_feature[i] += num_no_edge * np.eye(self.max_size)
            # self connection is added for each slice. Note that we do not model the diagonal part in flow.
            # this operation will make the diagonal of 4-th slices become 2. But we neither use diagonal of 4-th in gcn nor use it in flow update.

        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature)}
        
'''   
class PretrainZinkDataset_prop(Dataset):
    def __init__(self, node_features, adj_features, dts_features, dts_adj_features, mol_sizes, all_smiles, devide_mol, qeds, logps):
        self.n_molecule = node_features.shape[0]  # 249440
        self.node_features = node_features  # 2494400 * 38
        self.adj_features = adj_features  # 249440 * 3 * 38 * 38
        self.mol_sizes = mol_sizes  # 249440
        self.dts_features = dts_features  # 2494400 * 38
        self.dts_adj_features = dts_adj_features
        self.all_smiles = all_smiles
        self.devide_mol = devide_mol
        self.qeds = qeds
        self.logps = logps
        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.max_size = self.node_features.shape[1]  # 38
        self.node_dim = len(self.atom_list)  # 10

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy()  # (N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32)  # (3, N, N)
        dts_feature_copy = self.dts_features[idx].copy().astype(np.float32)  # (N)
        dts_adj_features = self.dts_adj_features[idx].copy().astype(np.float32)  # (N)
        qed_copy = self.qeds[idx].copy().astype(np.float32)  # (N)
        logps_copy = self.logps[idx].copy().astype(np.float32)  # (N)
        smiles = self.all_smiles[idx].copy()  # (N)
        devide_mol = self.devide_mol[idx].copy()  # (N)
        mol_size = self.mol_sizes[idx]

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size]  # (mol_size, mol_size)
        local_perm = np.random.permutation(mol_size)  # (first perm graph)生成一个随机原子序列
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]  # 按对应原子序列对 边矩阵重排
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0])  # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx))  # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)])

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        dts_feature_copy = dts_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        devide_mol = devide_mol[np.ix_(bfs_perm_origin)]  # (N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)

        for i in range(3):
            adj_feature_copy[i] = np.where(adj_feature_copy[i] > 0, dts_adj_features, 0)
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]

        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32)  # (N,10)
        for i in range(self.max_size):
            value_dts = dts_feature_copy[i]
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = value_dts
        node_feature = node_feature[:, :-1]  # (N, 9)
        # num_no_edge = adj_feature_copy[adj_feature_copy > 0].mean()

        t_adj = np.sum(adj_feature_copy, axis=0, keepdims=True)
        adj_ones = np.ones_like(t_adj)
        con_adj = np.where(t_adj > 0, adj_ones, 0)
        adj_feature = np.concatenate([adj_feature_copy, 1 - con_adj], axis=0).astype(np.float32)  # (4, N, N)
        # adj_feature[3] *= num_no_edge

        for i in range(4):
            adj_feature[i] += np.eye(self.max_size)
            # adj_feature[i] += num_no_edge * np.eye(self.max_size)
            # self connection is added for each slice. Note that we do not model the diagonal part in flow.
            # this operation will make the diagonal of 4-th slices become 2. But we neither use diagonal of 4-th in gcn nor use it in flow update.
        # print(devide_mol)
       return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature), 'raw_smile': smiles, 'devide_mol': devide_mol, 'qeds': qed_copy, 'plogp': logps_copy, 'mol_size': mol_size, 'bfs_perm_origin': bfs_perm_origin}
'''
class PretrainZinkDataset(Dataset):
    def __init__(self, node_features, adj_features, dts_features, dts_adj_features, mol_sizes, all_smiles, mol_segs):
        self.n_molecule = node_features.shape[0]  # 249440
        self.node_features = node_features  # 2494400 * 38
        self.adj_features = adj_features  # 249440 * 3 * 38 * 38
        self.mol_sizes = mol_sizes  # 249440
        self.dts_features = dts_features  # 2494400 * 38
        self.dts_adj_features = dts_adj_features
        self.all_smiles = all_smiles
        self.mol_segs = mol_segs
        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.max_size = self.node_features.shape[1]  # 38
        self.node_dim = len(self.atom_list)  # 10

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy()  # (N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32)  # (3, N, N)
        dts_feature_copy = self.dts_features[idx].copy().astype(np.float32)  # (N)
        dts_adj_features = self.dts_adj_features[idx].copy().astype(np.float32)  # (N)
        smiles = self.all_smiles[idx].copy()  # (N)
        mol_seg = self.mol_segs[idx].copy()  # (N)
        mol_size = self.mol_sizes[idx]

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size]
        local_perm = np.random.permutation(mol_size)  #
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0])  # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx))  # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)])

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        dts_feature_copy = dts_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)
        
        for i in range(3):
            adj_feature_copy[i] = np.where(adj_feature_copy[i] > 0, dts_adj_features, 0)
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]

        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32)  # (N,10)
        for i in range(self.max_size):
            value_dts = dts_feature_copy[i]
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = value_dts
        node_feature = node_feature[:, :-1]  # (N, 9)

        t_adj = np.sum(adj_feature_copy, axis=0, keepdims=True)
        adj_ones = np.ones_like(t_adj)
        con_adj = np.where(t_adj > 0, adj_ones, 0)
        adj_feature = np.concatenate([adj_feature_copy, 1 - con_adj], axis=0).astype(np.float32)  # (4, N, N)

        for i in range(4):
            adj_feature[i] += np.eye(self.max_size)

        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature), 'seg_mol': torch.Tensor(mol_seg)}


class PretrainMosesDataset(Dataset):
    def __init__(self, node_features, adj_features, dts_features, dts_adj_features, mol_sizes, all_smiles, devide_mol):
        self.n_molecule = node_features.shape[0]  # 249440
        self.node_features = node_features  # 2494400 * 38
        self.adj_features = adj_features  # 249440 * 3 * 38 * 38
        self.mol_sizes = mol_sizes  # 249440
        self.dts_features = dts_features  # 2494400 * 38
        self.dts_adj_features = dts_adj_features
        self.all_smiles = all_smiles
        self.devide_mol = devide_mol
        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 16, 17, 35, 0]
        self.max_size = self.node_features.shape[1]  # 38
        self.node_dim = len(self.atom_list)  # 10

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy()  # (N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32)  # (3, N, N)
        dts_feature_copy = self.dts_features[idx].copy().astype(np.float32)  # (N)
        dts_adj_features = self.dts_adj_features[idx].copy().astype(np.float32)  # (N)
        smiles = self.all_smiles[idx].copy()  # (N)
        devide_mol = self.devide_mol[idx].copy()  # (N)
        mol_size = self.mol_sizes[idx]

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size]  # (mol_size, mol_size)
        local_perm = np.random.permutation(mol_size)  # (first perm graph)生成一个随机原子序列
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]  # 按对应原子序列对 边矩阵重排
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0])  # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx))  # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)])

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        dts_feature_copy = dts_feature_copy[np.ix_(bfs_perm_origin)]  # (N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)

        for i in range(3):
            adj_feature_copy[i] = np.where(adj_feature_copy[i] > 0, dts_adj_features, 0)
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]

        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32)  # (N,10)
        for i in range(self.max_size):
            value_dts = dts_feature_copy[i]
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = value_dts
        node_feature = node_feature[:, :-1]  # (N, 9)
        # num_no_edge = adj_feature_copy[adj_feature_copy > 0].mean()

        t_adj = np.sum(adj_feature_copy, axis=0, keepdims=True)
        adj_ones = np.ones_like(t_adj)
        con_adj = np.where(t_adj > 0, adj_ones, 0)
        adj_feature = np.concatenate([adj_feature_copy, 1 - con_adj], axis=0).astype(np.float32)  # (4, N, N)
        # adj_feature[3] *= num_no_edge

        for i in range(4):
            adj_feature[i] += np.eye(self.max_size)

        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature), 'smiles': smiles, 'devide_mol': devide_mol}


class ConstrainOptim_Zink800(Dataset):
    def __init__(self, node_features, adj_features, mol_sizes, raw_smiles, plogps):        
        self.n_molecule = node_features.shape[0]
        self.node_features = node_features
        self.adj_features = adj_features
        self.mol_sizes = mol_sizes
        self.raw_smiles = raw_smiles
        self.plogps = plogps

        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.max_size = self.node_features.shape[1]
        self.node_dim = len(self.atom_list)

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy() #(N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32) #(3, N, N)
        mol_size = self.mol_sizes[idx]
        raw_smile = self.raw_smiles[idx]
        plogp = self.plogps[idx]

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size] #(mol_size, mol_size)
        local_perm = np.random.permutation(mol_size) #(first perm graph)
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0]) # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx)) # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)]) # maybe we should also provide bfs order #(N, )

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)] #(N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)
        for i in range(3):
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]
        
        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32) # (N,10)
        for i in range(self.max_size):
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = 1.
        node_feature = node_feature[:, :-1] #(N, 9)
        adj_feature = np.concatenate([adj_feature_copy, 1 - np.sum(adj_feature_copy, axis=0, keepdims=True)], axis=0).astype(np.float32) # (4, N, N)        
        for i in range(4):
            adj_feature[i] += np.eye(self.max_size) # self connection is added for each slice. Note that we do not model the diagonal part in flow.
            # this operation will make the diagonal of 4-th slices become 2. But we neither use diagonal of 4-th in gcn nor use it in flow update.
        
        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature), 
                'raw_smile': raw_smile, 'plogp': plogp, 'mol_size': mol_size, 'bfs_perm_origin': bfs_perm_origin}



class PositiveNegativeZinkDataset(Dataset):
    def __init__(self, node_features, adj_features, mol_sizes):        
        self.n_molecule = node_features.shape[0]
        self.node_features = node_features
        self.adj_features = adj_features
        self.mol_sizes = mol_sizes
        #self.labels = labels
        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.max_size = self.node_features.shape[1]
        self.node_dim = len(self.atom_list)

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy() #(N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32) #(3, N, N)
        mol_size = self.mol_sizes[idx]
        #label = self.labels[idx] # 1 indicate positive

        # get permutation and bfs
        pure_adj = np.sum(adj_feature_copy, axis=0)[:mol_size, :mol_size] #(mol_size, mol_size)
        local_perm = np.random.permutation(mol_size) #(first perm graph)
        adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
        adj_perm_matrix = np.asmatrix(adj_perm)
        G = nx.from_numpy_matrix(adj_perm_matrix)

        start_idx = np.random.randint(adj_perm.shape[0]) # operated on permed graph
        bfs_perm = np.array(bfs_seq(G, start_idx)) # get a bfs order of permed graph
        bfs_perm_origin = local_perm[bfs_perm]
        bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.max_size)])

        node_feature_copy = node_feature_copy[np.ix_(bfs_perm_origin)] #(N)
        perm_index = np.ix_(bfs_perm_origin, bfs_perm_origin)
        for i in range(3):
            adj_feature_copy[i] = adj_feature_copy[i][perm_index]
        
        node_feature = np.zeros((self.max_size, self.node_dim), dtype=np.float32) # (N,10)
        for i in range(self.max_size):
            index = self.atom_list.index(node_feature_copy[i])
            node_feature[i, index] = 1.
        node_feature = node_feature[:, :-1] #(N, 9)
        adj_feature = np.concatenate([adj_feature_copy, 1 - np.sum(adj_feature_copy, axis=0, keepdims=True)], axis=0).astype(np.float32) # (4, N, N)        
        for i in range(4):
            adj_feature[i] += np.eye(self.max_size) # self connection is added for each slice. Note that we do not model the diagonal part in flow.
            # this operation will make the diagonal of 4-th slices become 2. But we neither use diagonal of 4-th in gcn nor use it in flow update.
        
        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature)}




class DataIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        
    def __next__(self):
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data