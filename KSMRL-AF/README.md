# KSMRL Source Code

Note that the current code may contain bugs and is kind of messy, because we haven't reorganized the code due to the limited time.

## 1. Install Environment

* To install the rdkit, please refer to the official website. We highly recommend using anaconda3

  `conda create -c rdkit -n rdkit_env_test rdkit`

* Install torch, you may need to choose the correct version depending on the configuration of your machine

  `conda install pytorch torchvision cudatoolkit=11.1 -c pytorch`

## 2. Preprocess Dataset

We provide zinc250k dataset in `./dataset/250k_rndm_zinc_drugs_clean_sorted.smi`. 

- To preprocess zinc250 dataset. run `preprocess_data.sh`. The preprocessed features are stored in `./data_preprocessed`. 

## 3.Pretraining(Density Modeling and Generation)

* Run `train_parallel.sh`. 
* Checkpoints of each epochs will be save in `./save_pretrain`. 
* We will also generate 100 molecules at the end of each epoch and save these molecules in `./save_pretrain/model_name/mols	`
* Current implementation support multiple GPU for data parallel. You can set `CUDA_VISIBLE_DEVICES` in the script for gpu-parallel training.

## 4.Generation

* Given a checkpoint, you can generate molecules using the script `generate.sh`. 

* The generated molecules will be saved in `./mols`

  

Our implementation is based on GraphAF.  Thanks a lot for their awesome works.



