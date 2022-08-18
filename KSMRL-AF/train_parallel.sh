CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
--train --dataset zinc --num_workers 0 \
--batch_size 32 --edge_unroll 12 --lr 0.00005 --epochs 20 \
--shuffle --deq_coeff 0.9 --save --name 5-16-ksmrl-lamda_1 \
--num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
--is_bn --divide_loss --st_type exp --seed 2019 \
--all_save_prefix ./

CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zzz \
--train --dataset zinc --num_workers 0 \
--batch_size 32 --edge_unroll 12 --lr 0.00005 --epochs 20 \
--shuffle --deq_coeff 0.9 --save --name 5-16-ksmrl-lamda_1 \
--num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
--is_bn --divide_loss --st_type exp --seed 2019 \
--all_save_prefix ./
