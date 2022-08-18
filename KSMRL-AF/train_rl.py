#coding: utf-8
'''
Anonymous author
'''

from time import time
import argparse
import numpy as np
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model_rl import GraphFlowModel

import environment as env


def save_model(model, optimizer, args, var_list, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )    


def restore_model(model, args, epoch=None):
    if epoch is None:
        restore_path = os.path.join(args.save_path, 'checkpoint')
        print('restore from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('restore from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'])



class Trainer(object):

    def __init__(self, args):

        self.args = args                                 # 参数对象
        # 实例化图流模型  参数 38    9   4   12  参数对象
        self._model = GraphFlowModel(self.args)
        # 优化器对象     参数  模型参数    lr  weight_decay
        # 初始 loss 100
        # self.best_loss = 100.0
        # 初始 epoch 0
        # self.start_epoch = 0
        if self.args.cuda:
            self._model = self._model.cuda()

    # 加载已有的模型
    def initialize_from_checkpoint(self, gen=False):
        checkpoint = torch.load(self.args.init_checkpoint)
        # self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self._model.load_state_dict(checkpoint, strict=False)
        # if not gen:
        #     self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.best_loss = checkpoint['best_loss']
        #     self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)


    def train_prop_optim(self, lr, wd, max_iters, warm_up, pretrain_path, save_interval, save_dir):

        path = "./checkpoint12"
        # self.initialize_from_checkpoint(path)
        self._model.load_state_dict(torch.load(path))
        self._model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=lr, weight_decay=wd)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        print('start finetuning model(reinforce)')
        moving_baseline = None
        for cur_iter in range(max_iters):
            optimizer.zero_grad()
            loss, per_mol_reward, per_mol_property_score, moving_baseline = self._model.reinforce_forward_optim(
                in_baseline=moving_baseline, cur_iter=cur_iter)

            num_mol = len(per_mol_reward)
            avg_reward = sum(per_mol_reward) / num_mol
            avg_score = sum(per_mol_property_score) / num_mol
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self._model.flow_core.parameters()), 1.0)
            adjust_learning_rate(optimizer, cur_iter, lr, warm_up)
            optimizer.step()

            print('Iter {} | reward {}, score {}, loss {}'.format(cur_iter, avg_reward, avg_score, loss.item()))

            if cur_iter % save_interval == save_interval - 1:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'prop_opt_net_{}.pth'.format(cur_iter)))
        print("Finetuning (Reinforce) Finished!")


    def run_prop_optim(self, pretrain_path, n_mols=100, num_min_node=7, num_max_node=25,
                       temperature=0.75, atomic_num_list=[6, 7, 8, 9]):

        # self.get_model('prop_opt', model_conf_dict, checkpoint_path)
        self.initialize_from_checkpoint(pretrain_path)
        self._model.eval()
        all_mols, all_smiles = [], []
        cnt_mol = 0

        while cnt_mol < n_mols:
            mol, num_atoms = self._model.reinforce_optim_one_mol(atom_list=atomic_num_list, max_size_rl=num_max_node,
                                                                temperature=temperature)
            if mol is not None:
                smile = Chem.MolToSmiles(mol)
                if num_atoms >= num_min_node and not smile in all_smiles:
                    all_mols.append(mol)
                    all_smiles.append(smile)
                    cnt_mol += 1
                    if cnt_mol % 10 == 0:
                        print('Generated {} molecules'.format(cnt_mol))

        return all_mols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='zinc250k', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset')


    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')

    # ******model args******
    parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9, help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=6, help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    #TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='sigmoid', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******for exp st net only ******

    # ******for softplus st net only ******

    # ******optimization args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--train', action='store_true', default=True, help='do training.')
    parser.add_argument('--save', action='store_true', default=False, help='Save model.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

    parser.add_argument('--show_loss_step', type=int, default=100)

    # ******generation args******
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=5, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=48, help='maximum #atoms of generated mol')    
    parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--gen', action='store_true', default=False, help='generate')
    parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')
    parser.add_argument('--pretrainpath', type=str, default='./checkpoint12', help='output path for generated mol')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    trainer = Trainer(args)

    if args.train:

        trainer.train_prop_optim(0.0001, 0, 200, 0, args.pretrainpath, 20, "prop_opt_qed")

    else:

        trainer.run_prop_optim()


