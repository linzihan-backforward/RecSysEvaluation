# Evaluation Metric

## Experiment Setup

In this experiment,  we want to find the consistency between different metrics. 

We test the nine algorithms are tested on the eight datasets. Specially, for each dataset, we run all the comparison methods and obtain their performance scores, so that we can obtain a ranked list of comparison methods for each metric. Then, for each pair of two metrics, we compute the SRC and OR@5 between the corresponding two ranked lists. We repeat this process on each dataset, and average the ranking correlation scores as the correlation degree between two metrics. In this way, we can derive a correlation map among all pairs of evaluation metrics.

The detailed configuration for the rest factors are listed below. Yelp and Netflix datasets are processed with 10-core filtering as they are very large, and the other datasets are are processed with 10-core filtering. The interaction of each user is ordered by timestamp, and the entire dataset is split to train/validation/test sets by a ratio of 0.8/0/1/0.1. To optimize the model parameters, we employ the validation set for parameter search. To generate the candidate item list, we treat all the items that a user has not interacted with as candidates. All the metrics are computed based on a truncated length of 10, except AUC, which is computed on the whole list. 

## Running Steps

Run the following commands to get seven metrics, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

We take running `ItemKNN` on `ML-1M` dataset and `BPR` on `Netflix` as example,  other experiments' running steps are similar.

```bash
python run_recbole.py  --model=ItemKNN --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/ItemKNN.yaml' --metrics='["Recall", "MRR","NDCG","Hit","Precision","MAP","GAUC"]'

python run_recbole.py  --model=BPR --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/BPR.yaml' --metrics='["Recall", "MRR","NDCG","Hit","Precision","MAP","GAUC"]'
```









   

   

