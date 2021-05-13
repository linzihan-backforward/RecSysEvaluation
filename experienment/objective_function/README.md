# Objective Function

## Introduction

In this experiment,  we try the combination of different models and loss functions (i.e., BPR loss and BCE loss) and compare the performance of different combinations or methods.

To conduct the experiments, we categorize nine model into three categories according to their objective functions, including Non-sampling model (CDAE, MultiVAE and ENMF), BPR-based model (BPRMF, NGCF, LightGCN), and BCE-based model (NeuMF, SVD++ and FISM). Firstly, we aim to examine which category of recommendation algorithms generally have a better performance. Besides, since BPR and BCE losses share strong similarities, we aim to examine whether the choice between the BCE and BPR losses will lead to a substantial performance change in the models of the latter two categories.

The detailed configuration is available at `RecSysEvaluation/dataset/` directory. 

## Running Step

The loss function for origin `NGCF` is `BPR` and the name of its version with `BCE` Loss is `NGCFCE`, while the loss function for origin `NeuMF` is `BCE` and the name of its version with `BPR` Loss is `NeuMFBPR`. The same is true of other models.

Note that `CDAE, MultiVAE, ENMF` are non-sampling models, so the **`traing_neg_num` argument should be set to 0**.

Run the following commands, where `$Path` denotes the directory in which `RecSysEvaluation` is located.

In the following examples: 

- We run `BPR` with `BCE loss` on `ML-1M` dataset.
- We run `NeuMF` with `BPR loss` on `Netflix` dataset.
- We run `CDAE` with on `yelp` dataset.

Other experiments' running steps are similar.

```bash
python run_recbole.py --model=BPRCE --datasaet=ML-1M --config_files='$Path/RecSysEvaluation/dataset/ML-1M.yaml RecSysEvaluation/experiment/hyper_parameters/ML-1M/BPRCE.yaml'

python run_recbole.py --model=NeuMFBPR --datasaet=Netflix --config_files='$Path/RecSysEvaluation/dataset/Netflix.yaml RecSysEvaluation/experiment/hyper_parameters/Netflix/NeuMFBPR.yaml' 

python run_recbole.py --model=CDAE --datasaet=Yelp --config_files='$Path/RecSysEvaluation/dataset/Yelp.yaml RecSysEvaluation/experiment/hyper_parameters/Yelp/CDAE.yaml' --training_neg_num=0
```









   

   