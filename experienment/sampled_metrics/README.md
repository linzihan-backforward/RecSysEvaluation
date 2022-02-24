# Sampled Metrics

## Introduction

In this experiment,  we want to find how different candidate sets affect the model performance.

For these above purpose, we consider the two kinds of sampling methods, namely uniform sampling and popularity-based sampling , and use different numbers of irrelevant items in the set {100, 200, 500, 1000}. Given a combination of sampling method and negative number, we can form a candidate list of items for each dataset. We rank different algorithms according to their performance on both sampled ranking list and full-ranking list, and compute the SRC and HR@5 values between the two kinds of performance rankings. We repeat this process on each dataset, and report the average results. Moreover, we adopt a debiasing function on the sampled
metric to exam the performance under amendment for comparison, which is shown in the following equation.
$$
\hat{f}(\tilde{r})=f\left(1+\frac{(n-1)(r-1)}{m}\right)
$$
 where $r$ represents the ranking of positive instance on sampled candidates, $\tilde{r}$ is the correspondingly estimated ranking for whole candidates, and $m$ and $n$ denote the amount of candidates under sampling and no-sampling respectively. $f()$ represents any metric that calculate on the rankings.

Since previously we have found high correlations among some metrics, we only use three representative metrics (Recall@10, NDCG@10, AUC) for ranking. 

The detailed configuration is available at `RecSysEvaluation/dataset` directory . 

## Running Step

Run the following commands to get seven metrics, where `$Path` denotes the directory in which `RecSysEvaluation` is located. The sampling methods range in **`[uni100, uni200, uni500, uni1000, pop100, pop200, pop500, pop1000]`**.

In the following examples: 

- We run `ItemKNN` on `ML-1M` dataset and form a candidate set by sampling 100 irrelevant items for one relevant item uniformly.
- We run `BPR` on `Netflix` dataset and form a candidate set by sampling 1000 irrelevant items for one relevant item according to popularity.
- We run `NeuMF` on `LastFM` dataset and form a candidate set by sampling 200 irrelevant items for one relevant item uniformly. Note that it does not contain timestamps, **so we adopt `RO_RS` methods here**.

Other experiments' running steps are similar.

```bash
python run_recbole.py  --model=ItemKNN --dataset=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/sampled_metrics/hyper_parameters/ML-1M/ItemKNN.yaml' --eval_setting=TO_RS, uni100

python run_recbole.py  --model=BPR --dataset=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/sampled_metrics/hyper_parameters/Netflix/BPR.yaml' --eval_setting=TO_RS, pop1000

python run_recbole.py  --model=NeuMF --dataset=LastFM --config_files='$Path/RecSysEvaluation/dataset/LastFM.yaml RecSysEvaluation/experiment/sampled_metrics/hyper_parameters/LastFM/NeuMF.yaml' --eval_setting=RO_RS, uni200
```









   

   