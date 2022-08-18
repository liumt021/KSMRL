import os, torch, json, ast
import os.path as osp
import ssl
from itertools import repeat
import numpy as np
import tqdm
from rdkit import Chem
import pandas as pd
import sys
import networkx as nx
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url
import numpy as np
from rdkit import Chem
from dig.ggraph.dataset import PygDataset
from predata import cal_dis_info

bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}

zinc_atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
qm9_atom_list = [6, 7, 8, 9]


def prepross(data):
    num_max_node = data.adj.shape[-1]
    add_mask = torch.eye(num_max_node, dtype=torch.float32).unsqueeze(0)
    del_mask = abs(add_mask - 1.)
    adj_dis = data.adj_dis.unsqueeze(0)
    for i in range(data.x.shape[0] // num_max_node):
        data.x[i * num_max_node: (i + 1) * num_max_node, ...] = data.x[i * num_max_node: (i + 1) * num_max_node, ...] / data.x[i * num_max_node: (i + 1) * num_max_node, ...].max()
        data.x[i * num_max_node: (i + 1) * num_max_node, ...] = torch.where(data.x[i * num_max_node: (i + 1) * num_max_node, ...] > 0, 1 / torch.exp(data.x[i * num_max_node: (i + 1) * num_max_node, ...]), torch.zeros(1))
        data.x[i * num_max_node: (i + 1) * num_max_node, ...] = data.x[i * num_max_node: (i + 1) * num_max_node, ...] / data.x[i * num_max_node: (i + 1) * num_max_node, ...].max()

        data.adj[i * 4: (i + 1) * 4 - 1, ...] = data.adj[i * 4: (i + 1) * 4 - 1, ...] * adj_dis[:, i * num_max_node: (i + 1) * num_max_node, ...]
        data.adj[i * 4: (i + 1) * 4 - 1, ...] = data.adj[i * 4: (i + 1) * 4 - 1, ...] * del_mask
        data.adj[i * 4: (i + 1) * 4 - 1, ...] = data.adj[i * 4: (i + 1) * 4 - 1, ...] / data.adj[i * 4: (i + 1) * 4 - 1, ...].max()
        data.adj[i * 4: (i + 1) * 4 - 1, ...] = torch.where(data.adj[i * 4: (i + 1) * 4 - 1, ...] > 0, 1 / torch.exp(data.adj[i * 4: (i + 1) * 4 - 1, ...]), torch.zeros(1))
        data.adj[i * 4: (i + 1) * 4 - 1, ...] = data.adj[i * 4: (i + 1) * 4 - 1, ...] / data.adj[i * 4: (i + 1) * 4 - 1,...].max()
        data.adj[i * 4: (i + 1) * 4 - 1, ...] = data.adj[i * 4: (i + 1) * 4 - 1, ...] + add_mask
    data.adj_dis = None
    return data

class QM9(PygDataset):
    r"""A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset which is from `"MoleculeNet: A Benchmark for Molecular Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper and connsists of about 130,000 molecules with 2 property optimization targets: :obj:`penalized_logp` and :obj:`qed`.

    Args:
        root (string, optional): Root directory where the dataset should be saved.
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """

    def __init__(self,
                 root='./',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        name = 'qm9_property'
        super(QM9, self).__init__(root, name, prop_name, conf_dict,
                                  transform, pre_transform, pre_filter,
                                  processed_filename, use_aug, one_shot)

    def pre_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
            prop_list = list(input_df[self.prop_name])

        self.all_smiles = smile_list

        data_list = []

        for i, smile in enumerate(tqdm.tqdm(smile_list, ncols=100)):

            try:
                mol = Chem.MolFromSmiles(smile)
                x_dis, adj_dis = cal_dis_info(mol)
                
                x_dis_input = np.zeros(self.num_max_node, dtype=np.float32)
                x_dis_input[:len(x_dis)] = x_dis
                x_dis_input = x_dis_input[:, np.newaxis].repeat(len(self.atom_list), axis=1)
                
                adj_dis_input = np.zeros((self.num_max_node, self.num_max_node), dtype=np.float32)
                adj_dis_input[:len(x_dis), :len(x_dis)] = adj_dis

                Chem.Kekulize(mol)
                num_atom = mol.GetNumAtoms()
                if num_atom > self.num_max_node:
                    continue
                else:
                    # atoms
                    atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)

                    atom_idx = 0
                    for atom in mol.GetAtoms():
                        atom_feature = atom.GetAtomicNum()
                        atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1.
                        atom_idx += 1

                    atom_array = atom_array * x_dis_input
                    x = torch.tensor(atom_array)                # torch.Size([20, 4])

                    # bonds
                    adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                    for bond in mol.GetBonds():
                        bond_type = bond.GetBondType()
                        ch = bond_type_to_int[bond_type]
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        adj_array[ch, i, j] = 1.0
                        adj_array[ch, j, i] = 1.0
                                                            
                    adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)                                                           
                    adj_array += np.eye(self.num_max_node)              # (4, 20, 20)                    

                    data = Data(x=x)
                    data.adj = torch.tensor(adj_array)
                    data.adj_dis = torch.tensor(adj_dis_input)
                    data.num_atom = num_atom
                    
                    if self.available_prop:
                        data.y = torch.tensor([float(prop_list[i])])
                    data_list.append(data)
            except:
                continue

        data, slices = self.collate(data_list)
        data = prepross(data)
        del slices['adj_dis']
        return data, slices

class ZINC250k(PygDataset):
    r"""A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`ZINC250k` dataset which comes from the `ZINC database <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the `"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper and contains about 250,000 molecular graphs with up to 38 heavy atoms.

    Args:
        root (string, optional): Root directory where the dataset should be saved.
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)

    The dataset can be merged into a batch data format with :class:`torch_geometric.data.DataLoader` and :class:`torch_geometric.data.DenseDataLoader`. While :class:`DenseDataLoader` work with dense adjacency matrices and put batch information into an additional attribute :obj:`batch`, :class:`DataLoader` concatenate all graph attributes into one large graph. You can iterate over the data loader and see what it yields.

    Examples
    --------

    >>> dataset = ZINC250k(root='./dataset', prop_name='penalized_logp')
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> denseloader = DenseDataLoader(dataset, batch_size=32, shuffle=True)
    >>> data = next(iter(loader))
    >>> data
    Batch(adj=[128, 38, 38], batch=[1216], bfs_perm_origin=[1216], num_atom=[32], ptr=[33], smile=[32], x=[1216, 9], y=[32])
    >>> data = next(iter(denseloader))
    >>> data
    Batch(adj=[32, 4, 38, 38], bfs_perm_origin=[32, 38], num_atom=[32, 1], smile=[32], x=[32, 38, 9], y=[32, 1])

    Where the attributes of the output data indicates:

    * :obj:`x`: The node features.
    * :obj:`y`: The property labels for the graph.
    * :obj:`adj`: The edge features in the form of adjacent matrices.
    * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    * :obj:`bfs_perm_origin`: The bfs-searching order for single graph
    * :obj:`num_atom`: Number of atoms for each graph.
    * :obj:`smile`: Original SMILE sequences for the graphs.

    The dataset object is provided with training-validation split indices :obj:`get_split_idx()`, a list for all atom types :obj:`atom_list`, and the maximum number of nodes (atoms) among all molecules :obj:`num_max_node`.

    Examples
    --------

    >>> dataset.num_max_node
    38
    >>> dataset.atom_list
    [6, 7, 8, 9, 15, 16, 17, 35, 53]
    """

    def __init__(self,
                 root='./',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        name = 'zinc250k_property'
        super(ZINC250k, self).__init__(root, name, prop_name, conf_dict,
                                       transform, pre_transform, pre_filter,
                                       processed_filename, use_aug, one_shot)

    def pre_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
            prop_list = list(input_df[self.prop_name])

        self.all_smiles = smile_list

        data_list = []

        for i, smile in enumerate(tqdm.tqdm(smile_list, ncols=100)):

            try:
                mol = Chem.MolFromSmiles(smile)
                x_dis, adj_dis = cal_dis_info(mol)
                
                x_dis_input = np.zeros(self.num_max_node, dtype=np.float32)
                x_dis_input[:len(x_dis)] = x_dis
                x_dis_input = x_dis_input[:, np.newaxis].repeat(len(self.atom_list), axis=1)
                
                adj_dis_input = np.zeros((self.num_max_node, self.num_max_node), dtype=np.float32)
                adj_dis_input[:len(x_dis), :len(x_dis)] = adj_dis

                Chem.Kekulize(mol)
                num_atom = mol.GetNumAtoms()
                if num_atom > self.num_max_node:
                    continue
                else:
                    # atoms
                    atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)

                    atom_idx = 0
                    for atom in mol.GetAtoms():
                        atom_feature = atom.GetAtomicNum()
                        atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1.
                        atom_idx += 1

                    atom_array = atom_array * x_dis_input
                    x = torch.tensor(atom_array)

                    # bonds
                    adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                    for bond in mol.GetBonds():
                        bond_type = bond.GetBondType()
                        ch = bond_type_to_int[bond_type]
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        adj_array[ch, i, j] = 1.0
                        adj_array[ch, j, i] = 1.0
                                                            
                    adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)                                                           
                    adj_array += np.eye(self.num_max_node)              # (4, 38, 38)                    

                    data = Data(x=x)
                    data.adj = torch.tensor(adj_array)
                    data.adj_dis = torch.tensor(adj_dis_input)
                    data.num_atom = num_atom
                    
                    if self.available_prop:
                        data.y = torch.tensor([float(prop_list[i])])
                    data_list.append(data)
            except:
                continue

        data, slices = self.collate(data_list)
        data = prepross(data)
        del slices['adj_dis']
        return data, slices


class ZINC800(PygDataset):
    r"""A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`ZINC800` dataset which contains 800 selected molecules with lowest penalized logP scores. While method :obj:`jt` selects from the test set and :obj:`graphaf` selects from the train set.

    Args:
        root (string, optional): Root directory where the dataset should be saved.
        method (string, optional): Method name for :obj:`ZINC800` dataset, can be either :obj:`jt` or :obj:`graphaf`. (default: :obj:`jt`)
        prop_name (string, optional): The molecular property desired and used as the optimization target.(default: :obj:`penalized_logp`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """

    def __init__(self,
                 root='./',
                 method='jt',
                 prop_name='penalized_logp',
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):

        name = 'zinc800'

        if method in ['jt', 'graphaf']:
            name = 'zinc_800' + '_' + method
        else:
            error_mssg = 'Invalid method type {}.\n'.format(method)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(['jt', 'graphaf'])
            raise ValueError(error_mssg)
        super(ZINC800, self).__init__(root, name, prop_name, conf_dict,
                                      transform, pre_transform, pre_filter,
                                      processed_filename, use_aug, one_shot)


class MOSES(PygDataset):
    r"""A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`MOSES` dataset which is from the paper `"Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models" <https://arxiv.org/abs/1811.12823>`_ and contains 4,591,276 molecules refined from the ZINC database.

    Args:
        root (string, optional): Root directory where the dataset should be saved.
        prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
        one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
    """

    def __init__(self,
                 root='./',
                 prop_name=None,
                 conf_dict=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 use_aug=False,
                 one_shot=False
                 ):
        name = 'moses'
        super(MOSES, self).__init__(root, name, prop_name, conf_dict, transform, pre_transform, pre_filter,
                                    processed_filename, use_aug, one_shot)

    def pre_process(self):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', dtype='str')
        smile_list = list(input_df[self.smile_col])
        if self.available_prop:
            prop_list = list(input_df[self.prop_name])

        self.all_smiles = smile_list

        data_list = []

        for i, smile in enumerate(tqdm.tqdm(smile_list, ncols=100)):

            try:
                mol = Chem.MolFromSmiles(smile)
                x_dis, adj_dis = cal_dis_info(mol)
                
                x_dis_input = np.zeros(self.num_max_node, dtype=np.float32)
                x_dis_input[:len(x_dis)] = x_dis
                x_dis_input = x_dis_input[:, np.newaxis].repeat(len(self.atom_list), axis=1)
                
                adj_dis_input = np.zeros((self.num_max_node, self.num_max_node), dtype=np.float32)
                adj_dis_input[:len(x_dis), :len(x_dis)] = adj_dis

                Chem.Kekulize(mol)
                num_atom = mol.GetNumAtoms()
                if num_atom > self.num_max_node:
                    continue
                else:
                    # atoms
                    atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)

                    atom_idx = 0
                    for atom in mol.GetAtoms():
                        atom_feature = atom.GetAtomicNum()
                        atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1.
                        atom_idx += 1

                    atom_array = atom_array * x_dis_input
                    x = torch.tensor(atom_array)                # torch.Size([20, 4])

                    # bonds
                    adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
                    for bond in mol.GetBonds():
                        bond_type = bond.GetBondType()
                        ch = bond_type_to_int[bond_type]
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        adj_array[ch, i, j] = 1.0
                        adj_array[ch, j, i] = 1.0
                                                            
                    adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)                                                           
                    adj_array += np.eye(self.num_max_node)              # (4, 20, 20)                    

                    data = Data(x=x)
                    data.adj = torch.tensor(adj_array)
                    data.adj_dis = torch.tensor(adj_dis_input)
                    data.num_atom = num_atom
                    
                    if self.available_prop:
                        data.y = torch.tensor([float(prop_list[i])])
                    data_list.append(data)
            except:
                continue

        data, slices = self.collate(data_list)
        data = prepross(data)
        del slices['adj_dis']
        return data, slices
        
# if __name__ == '__main__':
#     test = QM9()
#     print(test[0])
#     import pdb; pdb.set_trace()

#     test = ZINC250k()
#     print(test[0])

#     test = ZINC800(method='jt')
#     print(test[0])

#     test = MOSES()
#     print(test[0])
