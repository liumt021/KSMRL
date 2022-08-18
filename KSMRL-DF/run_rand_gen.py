import json
import argparse
import sys
import os
from rdkit import RDLogger, Chem
from torch_geometric.data import DenseDataLoader
from Datasets import QM9, ZINC250k, MOSES
from dig.ggraph.method import GraphDF
from dig.ggraph.evaluation import RandGenEvaluator


RDLogger.DisableLog('rdApp.*')
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='qm9', choices=['qm9', 'zinc250k', 'moses'], help='dataset name')
parser.add_argument('--model_path', type=str, default='./saved_ckpts/rand_gen/rand_gen_qm9.pth',
                    help='The path to the saved model file')
parser.add_argument('--smi_path', type=str, default='./smi', help='The path to the saved smiles')
parser.add_argument('--eval_path', type=str, default='./eval', help='The path to the evaluation results')
parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
parser.add_argument('--train', action='store_true', default=False,
                    help='specify it to be true if you are running training')

args = parser.parse_args()

if args.data == 'qm9':
    with open('config/rand_gen_qm9_config_dict.json') as f:
        conf = json.load(f)
    dataset = QM9(conf_dict=conf['data'], one_shot=False, use_aug=True)
elif args.data == 'zinc250k':
    with open('config/rand_gen_zinc250k_config_dict.json') as f:
        conf = json.load(f)
    dataset = ZINC250k(conf_dict=conf['data'], one_shot=False, use_aug=True)
elif args.data == 'moses':
    with open('config/rand_gen_moses_config_dict.json') as f:
        conf = json.load(f)
    dataset = MOSES(conf_dict=conf['data'], one_shot=False, use_aug=True)
else:
    print("Only qm9, zinc250k and moses datasets are supported!")
    exit()

runner = GraphDF()

if args.train:
    loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
    runner.train_rand_gen(loader, conf['lr'], conf['weight_decay'], conf['max_epochs'], conf['model'],
                          conf['save_interval'], conf['save_dir'])
else:
    write_model_path = args.model_path.split('/')[-1]
    print(write_model_path)
    smiles_out = []
    mols, pure_valids = runner.run_rand_gen(conf['model'], args.model_path, args.num_mols, conf['num_min_node'],
                                            conf['num_max_node'], conf['temperature'], conf['atom_list'])
    for mol in mols:
        smiles_out.append(Chem.MolToSmiles(mol))

    if not os.path.exists(args.smi_path):
        os.makedirs(args.smi_path)
    output_smi_name = f'{args.data}-Num_{args.num_mols}.text'
    output_smi_path = os.path.join(args.smi_path, output_smi_name)
    fp = open(output_smi_path, 'w')
    fp.write(write_model_path + '\n')
    for i in range(len(smiles_out)):
        fp.write(smiles_out[i] + '\n')
    fp.close()

    smiles = [data.smile for data in dataset]
    evaluator = RandGenEvaluator()
    input_dict = {'mols': mols, 'train_smiles': smiles}

    print('Evaluating...')
    results = evaluator.eval(input_dict)
    print("Valid Ratio without valency check: {:.2f}%".format(sum(pure_valids) / args.num_mols * 100))


    if not os.path.exists(args.eval_path):
        os.makedirs(args.eval_path)
    output_smi_name = f'Eval.text'
    output_smi_path = os.path.join(args.smi_path, output_smi_name)
    fp1 = open(output_smi_path, 'w')
    fp1.write(write_model_path + '\n')
    for k, v in results.items():
        fp1.write(f"{k}: {v} \n")
    fp1.write('Valid Ratio without valency check:' + str(sum(pure_valids) / args.num_mols * 100) + '\n')
    fp1.close()