# Dataset Selection and Preprocessing

## Experiment Setup

In this experiment,  we want to find how different data processing methods affect the model performance.

To conduct the studies, for each dataset, we first consider applying one of two filtering strategies { 5-score, 10-core } to obtain the filtered datasets. Then, for each two filtering strategies, we compute and report their ranking correlation using the SRC measurement  with the original data without filtering. We report the results with the three representative metrics (Recall@10, NDCG@10, AUC) for ranking. 

The detailed configuration is available at `RecSysEvaluation/dataset` directory . 

## Running Step

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

In the following examples: 

- We run `ItemKNN` on `ML-1M` dataset processed with `5-core` filtering. 

- We run `BPR` on `Netflix` dataset processed with `10-core` filtering.

Other experiments' running steps are similar.

```bash
python run_recbole.py  --model=ItemKNN --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/ItemKNN.yaml' --min_user_inter_num=5 --min_item_inter_num=5

python run_recbole.py  --model=BPR --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/BPR.yaml' --min_user_inter_num=10 --min_item_inter_num=10
```









   

   