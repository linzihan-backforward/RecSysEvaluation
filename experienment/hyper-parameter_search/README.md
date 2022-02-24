# Objective Function

## Introduction

In this experiment, we try to compare the results of grid search with sequential search and find some empirical
findings to set hyper-parameters.

Here, we mainly consider studying the common hyper-parameters in neural recommendation algorithms, including the learning rate, embedding size, hidden layer size and regularization weight. These four hyper-parameters are typically used in different algorithms. If we could find some empirical findings to set them, it will largely reduce the efforts for parameter tuning. We remove the popularity and ItemKNN algorithms, since they are not optimization-based algorithms. Some algorithms might involve other parameters except the four parameters, and we still adopt the grid search approach to optimizing them. Since they are not our focus, we omit their results.

The detailed configuration is available at `RecSysEvaluation/dataset/` directory. 

## Running Step

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

In the following examples: 

- We run `BPR` on`ML-1M` dataset and find best hyper-parameters through `sequential search`.

  ```bash
  python run_hyper_obyo.py --model=BPR --dataset=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml' --params_file='$Path/RecSysEvaluation/experiment/hyper-parameter_search/hyper_parameters/ML-1M/searching_range/BPR.hyper' 
  ```

- We run `NeuMF` on`Netflix` dataset and find best hyper-parameters through `grid search`.

  ```bash
  python run_hyper.py --model=NeuMF --dataset=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml' --params_file='$Path/RecSysEvaluation/experiment/hyper-parameter_search/hyper_parameters/Netflix/searching_range/NeuMF.hyper' 
  ```

- We run `BPR` on`ML-1M` dataset and find best hyper-parameters through `sequential search` with `early stop` equal to 2.

  ```bash
  python run_hyper_obyo.py --model=BPR --dataset=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml' --params_file='$Path/RecSysEvaluation/experiment/hyper-parameter_search/hyper_parameters/ML-1M/searching_range/BPR.hyper' --early_stop=2
  ```

- We run `BPR` on `ML-1M` dataset with best hyper-parameters searched by `sequential search`.

  ```bash
  python run_recbole.py --model=BPR --dataset=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml  $Path/RecSysEvaluation/experiment/hyper-parameter_search/hyper_parameters/ML-1M/sequential_search/BPR.yaml' 
  ```

- We run `NeuMF` on`Netflix` dataset with best hyper-parameters searched by `grid search`.

  ```bash
  python run_recbole.py --model=NeuMF --dataset=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml  $Path/RecSysEvaluation/experiment/hyper-parameter_search/hyper_parameters/Netflix/grid_search/NeuMF.yaml' 
  ```

Other experiments' running steps are similar.

## Hyperparameters Results

The following tables contains the searching range and optimal hyperparameters in `ML-1M` and `Netflix` got from `grid search`. The meaning of specific hyper-parameter can be found in the documents of our framework.

### ML-1M

| Model    | Optimal Hyperparameters                                      | Searching Range                                              |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BPR      | embedding_size=2048<br/>learning_rate=1e-4                   | embedding_size in [16,32,64,128,256,512,<br/>1024,2048,4096]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,<br/>5e-3,1e-2,5e-2] |
| SVD++    | embedding_size=128<br/>learning_rate=5e-4<br/>reg_weight=5e-4 | embedding_size in [64,128,256,512]<br/>learning_rate in [1e-4,5e-4,1e-3,5e-3,1e-2]<br/>reg_weight in [0,5e-5,5e-4,5e-3,5e-2] |
| NeuMF    | dropout_prob=0.3<br/>embedding_size=64<br/>learning_rate=5e-4<br/>mlp=[256,256,256] | dropout_prob in [0.1,0.2,0.3,0.5]<br/>embedding_size in [32,64,128,256,512]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3]<br/>mlp in [[256,128,256],[128,128,128]<br/>,[256,256,256],[64,64,64]] |
| NAIS     | embedding_size=32<br/>learning_rate=5e-5<br/>reg_weights=0   | embedding_size in [16,32,64,128]<br/>learning_rate in [1e-5,5e-5,1e-4,5e-4,1e-3]<br/>reg_weights in [1e-7,1e-3,0,10] |
| FISM     | embedding_size=128<br/>learning_rate=1e-3<br/>reg_weights=1e-3 | embedding_size in [64,128,256]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3]<br/>reg_weights in [1e-7,0,1e-1,1e-3,1e-5] |
| NGCF     | embedding_size=2048<br/>hidden_size=256<br/>learning_rate=5e-4<br/>reg_weight=1e-6 | embedding_size in [512,1024,2048]<br/>hidden_size in [256,512,1024]<br/>learning_rate in [1e-4,5e-4,1e-3,5e-3]<br/>reg_weight in [0,1e-6,1e-4,1e-2] |
| LightGCN | embedding_size=1024<br/>learning_rate=5e-4<br/>reg_weight=1e-2 | embedding_size in [256,512,1024,2048]<br/>learning_rate in [1e-4,5e-4,1e-3,5e-3,1e-2]<br/>reg_weight in [1e-6,1e-5,1e-4,1e-2,0] |
| ENMF     | dropout_prob=0.3<br/>embedding_size=256<br/>learning_rate=1e-3<br/>negative_weight=0.5 | dropout_prob in [0.5,0.3,0.1]<br/>embedding_size in [64,256,512]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]<br/>negative_weight in [5e-3,5e-2,0.5] |
| CDAE     | embedding_size=4096<br/>learning_rate=5e-3                   | embedding_size in [16,32,64,128,256,512,<br/>1024,2048,4096]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,<br/>1e-2,5e-2] |
| MultiVAE | latent_dimension=128<br/>learning_rate=1e-2<br/>mlp_hidden_size=600 | latent_dimension in [128,512,1024,2048]<br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]<br/>mlp_hidden_size in [100,300,600,1200,2400] |

### Netflix

| Model    | Optimal Hyperparameters                                      | Searching Range                                              |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BPR      | embedding_size=4096<br/>learning_rate=0.0001                 | embedding_size in [16,32,64,128,256,512,1024,2048,4096] <br>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2] |
| SVD++    | embedding_size=256<br/>learning_rate=0.0005                  | reg_weight in [0,5e-5,5e-4,5e-3,5e-2] <br/>embedding_size in [32,64,128,256] <br/>learning_rate in [1e-4,5e-4,1e-3,5e-3,1e-2] |
| NeuMF    | dropout_prob=0.2<br/>embedding_size=64<br/>learning_rate=5e-05<br/>mlp_hidden_size=[256,256,256] | dropout_prob in [0.2,0.3] <br/>mlp_hidden_size in ['[128,128,128]','[256,128,256]','[256,256,256]'] <br/>embedding_size in [64,128]<br/> learning_rate in [5e-5,1e-4,5e-4,1e-3] |
| NAIS     | alpha=0<br/>beta=0.5<br/>embedding_size=64<br/>learning_rate=0.0005<br/>reg_weights=[1e-3,1e-3,1e-3]<br/>weight_size=32} | weight_size in [32] <br/>alpha in [0] <br/>beta in [0.5] <br/>reg_weights in ['[1e-7,1e-7,1e-7]','[1e-3,1e-3,1e-3]','[0,0,0]'] <br/>embedding_size in [32,64] <br/>learning_rate in [1e-5,5e-5,1e-4,5e-4,1e-3] |
| FISM     | alpha=0<br/>beta=0.5<br/>embedding_size=64<br/>learning_rate=0.0005<br/>reg_weights=[1e-7,1e-7] | alpha in [0] <br/> reg_weights in ['[1e-3,1e-3]','[1e-5,1e-5]','[1e-7,1e-7]','[0,0]'] <br/>embedding_size in [32,64] <br/>learning_rate in [5e-4,1e-3,5e-3,1e-2] |
| NGCF     | embedding_size=128<br/>hidden_size_list='[64,64,64]'<br/>learning_rate=0.001<br/>message_dropout=0.1<br/>node_dropout=0.0<br/>reg_weight=0.01 | node_dropout in [0.0] <br/>message_dropout in [0.1] <br/>hidden_size_list in ['[256,256,256]','[64,64,64]'] <br/>reg_weight in [1e-6,1e-4,1e-2,0] <br/>embedding_size in [64,128,256] <br/>learning_rate in [1e-4,5e-4,1e-3,5e-3] |
| LightGCN | embedding_size=256<br/>learning_rate=0.0005<br/>n_layers=2<br/>reg_weight=0.01 | n_layers in [2] <br/>reg_weight in [1e-6,1e-5,0,1e-2,1e-4] <br/>embedding_size in [64,128,256]<br/> learning_rate in [1e-4,5e-4,1e-3,5e-3,1e-2] |
| ENMF     | dropout_prob=0.1<br/>embedding_size=64<br/>learning_rate=0.0005<br/>negative_weight=0.005 | dropout_prob in [0.5,0.3,0.1] <br/>negative_weight in [5e-3,5e-2,0.5] <br/>embedding_size in [32,64] <br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2] |
| CDAE     | embedding_size=256<br/>learning_rate=0.005<br/>reg_weight_1=0.01<br/>reg_weight_2=0.01 | reg_weight_1 in [0,0.01]<br/>reg_weight_2 in [0,0.01] <br/>embedding_size in [16,32,64,128,256] <br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2] |
| MuitiVAE | anneal_cap=0.1<br/>dropout_prob=0.5<br/>latent_dimension=2048<br/>learning_rate=0.005<br/>mlp_hidden_size=[1200]<br/>total_anneal_steps=200000} | total_anneal_steps in [200000] <br/>dropout_prob in [0.3,0.5] <br/>anneal_cap in [0.1,0.2] <br/>mlp_hidden_size in ['[100]','[300]','[600]','[1200]','[2400]'] <br/>latent_dimension in [128,512,1024,2048] <br/>learning_rate in [5e-5,1e-4,5e-4,1e-3,5e-3] |



   

   