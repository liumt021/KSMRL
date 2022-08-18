# train:
CUDA_VISIBLE_DEVICES=0 python run_rand_gen.py --train --data=zinc250k
# generate:
CUDA_VISIBLE_DEVICES=0 python run_rand_gen.py --num_mols=100 --model_path='./rand_gen_zinc250k.pth' --data=zinc250k --smi_path='./smi/ori.txt'

