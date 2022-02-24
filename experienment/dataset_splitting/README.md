# Data Splitting

## Introduction

In this experiment,  we want to find how splitting methods affect the model performance and whether the recommended items are valid (i.e. , whether the first time a recommended item appears is after the time of the last interaction from the corresponding user).

For the first question, we rank the list of comparison models given a splitting method. And, we measure the rank correlations between two splitting methods. The process is repeated for all the datasets except Last.FM dataset which does not contain timestamps. Besides, we mainly focus on the number of invalid recommendations that violate the constraints of global timing, which is formally computed as follows:
$$
\text { #invalid }=\sum_{u \in U} \sum_{i \in \hat{R}_{u}} I\left(T_{i}^{f} \geq T_{u}^{l}\right)
$$
where $\in \hat{R}_{u}$ denotes a recommendation list for user $u$ by some method, $T_{i}^{f}$ is the first time that item appears, $T_{u}^{l}$ is the time of the last interaction from user $u$, and $I(\cdot)$ is an indicator function that returns 1 when the condition is true and 0 otherwise.

## Running Step

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located. The splitting methods range in **`[RO_RS, RO_LS, TO_RS, TO_LS, GT_RS]`**.The meanings of them can be found in the documents of our framework.

In the following examples: 

- We run `ItemKNN` on `ML-1M` dataset which adopts **random ordering and ratio-based splitting** `(RO_RS)`.

- We run `BPR` on `Netflix` dataset which adopts **global temporal ordering and ratio-based splitting** `(GT_RS)`.

Other experiments' running steps are similar.

```bash
python run_recbole.py --model=ItemKNN --dataset=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/dataset_splitting/hyper_parameters/ML-1M/RO_RS/ItemKNN.yaml' --eval_setting=RO_RS,full

python run_recbole.py --model=BPR --dataset=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/dataset_splitting/hyper_parameters/Netflix/GT_RS/BPR.yaml' --eval_setting=TO_RS,full --group_by_user=False
```









   

   