CUDA_VISIBLE_DEVICES=0 python -u -W ignore constrain_optim_rl.py --path ./data_preprocessed/zinc_800 \
    --co_rl --save --num_workers 2 \
    --batch_size 64 --lr 0.0001 --epochs 1000 \
    --shuffle --deq_coeff 0.9 --name rl_co_iter300_rewardimp_nobaseline \
    --num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
    --is_bn --divide_loss --st_type exp \
    --init_checkpoint ./good_ckpt/checkpoint277 \
    --min_atoms 10 --plogp_coeff 0.33333 --exp_temperature 3.0 --reward_decay 0.9 \
    --reward_type imp \
    --seed 666 --max_atoms 38 --moving_coeff 0.99 --rl_sample_temperature 0.75 \
    --modify_size 5 --warm_up 24 --update_iters 4 \
    --reinforce_iters 300 --not_save_demon --no_baseline \
    --reinforce_fintune



CUDA_VISIBLE_DEVICES=0 python -u -W ignore constrain_optim_rl.py --path ./data_preprocessed/zinc250k_clean_sorted \
    --co_rl --save --num_workers 2 \
    --batch_size 64 --lr 0.0001 --epochs 1000 \
    --shuffle --deq_coeff 0.9 --name rl_co_iter300_rewardimp_nobaseline \
    --num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
    --is_bn --divide_loss --st_type exp \
    --init_checkpoint ./save_pretrain/exp_zinc_s998-7-18-epoch20/checkpoint12 \
    --min_atoms 10 --plogp_coeff 0.33333 --exp_temperature 3.0 --reward_decay 0.9 \
    --reward_type imp \
    --seed 666 --max_atoms 38 --moving_coeff 0.99 --rl_sample_temperature 0.75 \
    --modify_size 5 --warm_up 24 --update_iters 4 \
    --reinforce_iters 300 --not_save_demon --no_baseline \
    --reinforce_fintune