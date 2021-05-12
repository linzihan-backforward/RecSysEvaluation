# Sampled Metrics

## Experiment Setup

In this experiment,  we want to find how different candidate sets affect the model performance.

For these above purpose, we consider the two kinds of sampling methods, namely uniform sampling and popularity-based sampling , and use different numbers of irrelevant items in the set {100, 200, 500, 1000}. Given a combination of sampling method and negative number, we can form a candidate list of items for each dataset. We rank different algorithms according to their performance on both sampled ranking list and full-ranking list, and compute the SRC and HR@5 values between the two kinds of performance rankings. We repeat this process on each dataset, and report the average results. Moreover, we adopt a debiasing function on the sampled
metric to exam the performance under amendment for comparison. Since previously we have found high correlations among some metrics, we only use three representative metrics (Recall@10, NDCG@10, AUC) for ranking. 

The detailed configuration is available at `RecSysEvaluation/dataset` directory . 

## Running Step

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

In the following examples: 

- We run `ItemKNN` on `ML-1M` dataset and form a candidate set by sampling 100 irrelevant items for one relevant item uniformly.

- We run `BPR` on `Netflix` dataset and form a candidate set by sampling 1000 irrelevant items for one relevant item according to popularity.

Other experiments' running steps are similar.

```bash
python run_recbole.py  --model=ItemKNN --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/ItemKNN.yaml' --eval_setting=TO_RS, uni100

python run_recbole.py  --model=BPR --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/BPR.yaml' --eval_setting=TO_RS, pop1000
```









   

   