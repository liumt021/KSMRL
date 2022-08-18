# KSMRL-DF

Note that the current code may contain bugs and is kind of messy, because we haven't reorganized the code due to the limited time.



###  Train

```shell script
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --train --data=qm9 
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --train --data=zinc250k
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --train --data=mosesshell script
```
### Generation

```shell script
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --num_mols=100 --model_path=${path_to_the_model} --data=qm9
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --num_mols=100 --model_path=${path_to_the_model} --data=zinc250k
$ CUDA_VISIBLE_DEVICES=${your_gpu_id} python run_rand_gen.py --num_mols=100 --model_path=${path_to_the_model} --data=moses
```

### Acknowledgement

Our implementation is based on GraphDF. Thanks a lot for their awesome works.
