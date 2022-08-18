CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
--gen --gen_out_path ./mols/perfect.txt \
--batch_size 32 --lr 0.00005 --epochs 100 \
--shuffle --deq_coeff 0.9 --save --name l12_h128_o128_exp_sbatch \
--num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
--is_bn --divide_loss --st_type exp \
--init_checkpoint ./save_pretrain/exp_zinc_s998-7-18-epoch20/checkpoint12 \
--gen_num 10000 --min_atoms 8 --max_atoms 48 --save --seed 66666666 --temperature 0.75

#--gen_num 500 --min_atoms 10 --max_atoms 10 --save --seed 66666666 --temperature 0.7


CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
--gen --gen_out_path ./mols/test_100mol.txt \
--batch_size 32 --lr 0.001 --epochs 100 \
--shuffle --deq_coeff 0.9 --save --name l12_h128_o128_exp_sbatch \
--num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
--is_bn --divide_loss --st_type exp \
--init_checkpoint ./good_ckpt/checkpoint_co \
--gen_num 100 --min_atoms 10 --save --seed 66666666 --temperature 0.7


./save_pretrain/exp_zinc_s998-7-18-epoch20/checkpoint12
./prop_opt_net_99

CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
--gen --gen_out_path ./mols/perfect.txt \
--batch_size 32 --lr 0.00005 --epochs 100 \
--shuffle --deq_coeff 0.9 --save --name l12_h128_o128_exp_sbatch \
--num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
--is_bn --divide_loss --st_type exp \
--init_checkpoint ./prop_opt_net_1.pth \
--gen_num 10000 --min_atoms 20 --max_atoms 42 --save --seed 66666666 --temperature 0.75