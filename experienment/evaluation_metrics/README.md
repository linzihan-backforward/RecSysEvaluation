# Evaluation Metric

## Introduction

In this experiment,  we want to find the consistency between different metrics. 

We test the nine algorithms are tested on the eight datasets. Specially, for each dataset, we run all the comparison methods and obtain their performance scores, so that we can obtain a ranked list of comparison methods for each metric. Then, for each pair of two metrics, we compute the SRC and OR@5 between the corresponding two ranked lists. We repeat this process on each dataset, and average the ranking correlation scores as the correlation degree between two metrics. In this way, we can derive a correlation map among all pairs of evaluation metrics.

The detailed configuration is available at `RecSysEvaluation/dataset/` directory. We calculate seven metrics ( i.e. `Recall@10, MRR@10, NDCG@10, Hit@10, Precision@10, GAUC`) and try to find consistency between them.

## Running Step

Run the following commands to get seven metrics, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

We take running `ItemKNN` on `ML-1M` dataset and `BPR` on `Netflix` as example,  other experiments' running steps are similar.

```bash
python run_recbole.py  --model=ItemKNN --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/ItemKNN.yaml' --metrics='["Recall", "MRR","NDCG","Hit","Precision","MAP","GAUC"]'

python run_recbole.py  --model=BPR --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/BPR.yaml' --metrics='["Recall", "MRR","NDCG","Hit","Precision","MAP","GAUC"]'
```









   

   

