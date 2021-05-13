# Objective Function

## Introduction

In this experiment, we try to compare the results of grid search with sequential search and find some empirical
findings to set hyper-parameters.

Here, we mainly consider studying the common hyper-parameters in neural recommendation algorithms, including the learning rate, embedding size, hidden layer size and regularization weight. These four hyper-parameters are typically used in different algorithms. If we could find some empirical findings to set them, it will largely reduce the efforts for parameter tuning. We remove the popularity and ItemKNN algorithms, since they are not optimization-based algorithms. Some algorithms might involve other parameters except the four parameters, and we still adopt the grid search approach to optimizing them. Since they are not our focus, we omit their results.

The detailed configuration is available at `RecSysEvaluation/dataset/` directory. 

## Running Step

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

In the following examples: 

- We run `BPR` on `ML-1M` dataset and find best hyper-parameters by `grid search`.

- We run `NeuMF` on`Netflix` dataset and find best hyper-parameters by `sequential search`.

Other experiments' running steps are similar.

```bash
python run_hyper.py --model=BPR --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/BPR.yaml' --params_file='RecSysEvaluation/experiment/hyper_parameters/ML-1M/bpr.hyper'

python run_obyo.py --model=NeuMF --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/NeuMF.yaml' --params_file='RecSysEvaluation/experiment/hyper_parameters/Netflix/neumf.hyper'
```

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

| Model    | Optimal Hyperparameters | Searching Range |
| -------- | ----------------------- | --------------- |
| BPR      |                         |                 |
| SVD++    |                         |                 |
| NeuMF    |                         |                 |
| NAIS     |                         |                 |
| FISM     |                         |                 |
| NGCF     |                         |                 |
| LightGCN |                         |                 |
| ENMF     |                         |                 |
| CDAE     |                         |                 |
| MuitiVAE |                         |                 |



   

   