# RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data
Maxwell A Xu<sup>1,2*</sup>, Jaya Narain<sup>1</sup>, Gregory Darnell<sup>1</sup>, Haraldur T Hallgrimsson<sup>1</sup>, Hyewon Jeong<sup>1,3*</sup>, Darren Forde<sup>1</sup>, Richard Andres Fineman<sup>1</sup>, Karthik Jayaraman Raghuram<sup>1</sup>, James Matthew Rehg<sup>2</sup>, Shirley You Ren<sup>1</sup> 

<sub><sup>1</sup>Apple Inc.
<sup>2</sup>UIUC
<sup>3</sup>MIT &nbsp; &nbsp; | &nbsp; &nbsp;  <sup>*</sup>Work done while at Apple </sub>




####   Accepted at ICLR 2025. Please read our paper here: [https://arxiv.org/abs/2311.00519](https://arxiv.org/abs/2411.18822).


## Introduction
We present RelCon, a novel self-supervised Relative Contrastive learning approach for training a motion foundation model from wearable accelerometry sensors. First, a learnable distance measure is trained to capture motif similarity and domain-specific semantic information such as rotation invariance. Then, the learned distance provides a measurement of semantic similarity between a pair of accelerometry time-series, which we use to train our foundation model to model relative relationships across time and across subjects. The foundation model is trained on 1 billion segments from 87,376 participants, and achieves state-of-the-art performance across multiple downstream tasks, including human activity recognition and gait metric regression. To our knowledge, we are the first to show the generalizability of a foundation model with motion data from wearables across distinct evaluation tasks.



This codebase includes the full RelCon methodology and pre-training pipeline, along with data preprocessing and evaluation scripts for the public benchmarks presented in the paper. While we cannot release the pre-training dataset or the associated model weights, this repository provides the complete pre-training pipeline, enabling researchers to retrain the RelCon model on their own datasets. We hope this release will help construct a unified benchmark for motion tasks and also help facilitate further research in self-supervised learning for time-series foundation models and wearable data analysis

## Code Overview
Below is an outline of the overall structure of our codebase. The code is nicely modularized with modular class-based configs that help define specific components of an experiment, such as a config for tuning the model training or a config for designing the network backbone. Extending this codebase to your own use-cases should be fairly straightforward.
                    
```
run_exp.py           # Main file used to launch experiments  
relcon/              # Source code  
├── experiments/      
│   └── configs/     # Config for defining experiment
├── models/          # Training pipeline
│   └── RelCon/      # *** RelCon trainer ***
│   └── MotifDist/  
├── nets/            # Network backbones (e.g. ResNet)  
├── data/            
│   └── process/     # Downloading and preprocessing data  
└── eval/            # Evaluation pipeline  
```
## Code Usage

### (A) Python Environment

For this project we use [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) to manage dependencies. After installing miniconda, we can install the relcon environment with the following terminal commands:

    conda env create -f env.yml
    conda activate relcon
    pip install -e .

### (B) Data Download and Pre-processing
#### Pre-training Datasets
We can then generate a synthetic dataset for the RelCon pre-training pipeline that mimics the structure of the original pre-training dataset by running the below line. If you would like to pre-train the RelCon model on your own dataset, please pre-process the data so that it follows the structure described in `process_dummydataset.py`

    python relcon/data/process/process_dummydataset.py

#### Evaluation Datasets
The [Opportunity](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition), [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition), and [Motionsense](https://github.com/mmalekzadeh/motion-sense) datasets and splits can be downloaded and pre-processed with the below commands. See "Section 4.4.2 Benchmarking Evaluation Set-up" in the paper for more details.

    python relcon/data/process/process_comparepriorpt_datasets.py
    python relcon/data/process/process_comparesslbench_datasets.py

### (C) Training and Evaluating RelCon

### [Step 1] Training the Learnable Distance Measure

We train a neural network to **learn a distance measure to identify** whether two sequences have **similar temporal motifs** and are **semantically similar**. After training, the architecture is frozen and used as a static function **to determine the relative similarities of candidate samples** in the RelCon approach. 

<!-- The distance measure is defined below:

$$
\begin{align}
    d(\textbf{X}_{\textrm{anc}},\textbf{X}_{\textrm{cand}}) &\coloneqq \Vert \hat{\textbf{X}}_{\textrm{anc} \vert \textrm{cand}} -\textbf{X}_{\textrm{anc}}\Vert_2^2  \\
    \hat{\textbf{X}}_{\textrm{anc} \vert \textrm{cand}} &= \big( (\textrm{CrossAttn}({\textbf{X}}_\textrm{anc} \vert {\textbf{X}}_\textrm{cand}) \textbf{W}_o + \textbf{b}_o )  + {\mu}_{\textrm{cand}} \big) {\sigma}_{\textrm{cand}} \\
    \textrm{CrossAttn}({\textbf{x}}_\textrm{anc} \vert {\textbf{X}}_\textrm{cand}) 
    &= \sum_{\textbf{x}_\textrm{cand} \in {\textbf{X}}_\textrm{cand}} \underset{{\textbf{x}_\textrm{cand} \in {\textbf{X}}_\textrm{cand}}}{\textrm{sparsemax}} \Big( \textrm{sim}\big( f_q({\textbf{x}}_{\textrm{anc}}), f_k(\textbf{x}_{\textrm{cand}}) \big) f_v(\textbf{x}_{\textrm{cand}})\Big) \\

    f_{\{q/k/v\}}({\textbf{X}}_{\{\textrm{anc}/\textrm{cand}\}}) &= \textrm{DilatedConvNet}_{\{q/k/v\}} \Big( \frac{{\textbf{X}}_{\{\textrm{anc}/\textrm{cand}\}} - {\mu}_{\textrm{cand}}}{{\sigma}_{\textrm{cand}}} \Big)
\end{align}
$$

where $\smash{\textbf{X} \in \mathbb{R}^{T \times D}}$ and $\smash{\textbf{x}, \mu, \sigma \in \mathbb{R}^{D}}$ with $T$ as the time length and $D=3$ for our 3-axis accelerometry signals. The distance between an anchor sequence and a candidate sequence, $\smash{d(\textbf{X}_{\textrm{anc}},\textbf{X}_{\textrm{cand}})}$, is defined as the reconstruction accuracy to generate the anchor from the candidate. The distance measure is strictly dependent on the motif similarities between the anchor and candidate that are captured in the dilated convolutions in $\smash{f_{\{q/k\}}}$.  -->

The training pipeline can be found in `models/MotifDist/MotifDist_Model.py` and it uses the `nets/CrossAttn/CrossAttn_Net.py` network. The distance function can be trained by running:

    python run_exp.py --config 25_3_8_motifdist

`--config` refers to the specific dictionary str key associated with a given config that includes all experimental parameters, such as epochs or learning rate. Specifically `25_3_8_motifdist` originates from `experiments/configs/MotifDist_expconfigs.py`. All available configs can be found in `experiments/configs/`

### [Step 2] Training and Evaluating RelCon <sub> <sup> where all Pairs are Positive ... but Some Pairs are More Positive than Others.</sup> </sub>

The RelCon approach now takes the previously trained learnable distance measure and uses it to identify which candidates are more positive and which are more negative. RelCon then learns an **embedding space** that preserves the **hierarchical structure of relative similarities** between all of its candidates. This allows the model to **capture the subtle semantic differences between similar but distinct motions**, such as distinguishing between walking and running, or between indoor and outdoor cycling.


<!-- $$   
\begin{align*}
    \mathcal{L}_{\textrm{RelCon}} &= \sum_{\textbf{X}_{\textrm{i}} \in \mathcal{S}_{\textrm{cand}}} \ell(\textbf{X}_{\textrm{anc}},\ \textbf{X}_{pos} \coloneqq \textbf{X}_{\textrm{i}},\ \mathcal{S}_{\textrm{neg}} \coloneqq f_{\textrm{neg}}(\textbf{X}_{\textrm{anc}}, \textbf{X}_{\textrm{i}}, \mathcal{S}_{\textrm{cand}})) \\ 
    f_{\textrm{neg}}(\textbf{X}_{\textrm{anc}}, \textbf{X}_{\textrm{pos}}, \mathcal{S}) &= \{\textbf{X} \in \mathcal{S} : d(\textbf{X}_{\textrm{anc}}, \textbf{X}) > d(\textbf{X}_{\textrm{anc}}, \textbf{X}_{\textrm{pos}}) \}  \\
    \ell(\textbf{X}_{\textrm{anc}}, \textbf{X}_{\textrm{pos}}, \mathcal{S}_{\textrm{neg}})  &= - \log \frac{\exp(\textrm{sim}(\textbf{X}_{\textrm{anc}},  \textbf{X}_{\textrm{pos}}) / \tau)}{\sum_{\textbf{X}_{\textrm{neg}} \in \mathcal{S}_{\textrm{neg}}} \exp(\textrm{sim}(\textbf{X}_{\textrm{anc}},  \textbf{X}_{\textrm{neg}}) / \tau) + \exp(\textrm{sim}(\textbf{X}_{\textrm{anc}},  \textbf{X}_{\textrm{pos}}) / \tau)} 
\end{align*}
$$ -->

The training pipeline and relative loss function can be found in `models/RelCon/RelCon_Model.py`. RelCon is a network agnostic approach, but in our implementation, we use a ResNet backbone to encode the signals, which can be found in `nets/ResNet1D/ResNet1D_Net.py`. RelCon can be trained by running:

    python run_exp.py --config 25_3_8_relcon

Note that `25_3_8_relcon` also contains the evaluation configs as well, for the Benchmarking Evaluation tasks, which can be seen in Lines 64-141 in `experiments/configs/RelCon_expconfigs.py`. As such, after training, the evaluation pipeline will be ran with the pre-trained RelCon checkpoints. 


## Citation
If you use our work in your research, please cite

```bibtex
@article{xu2024relcon,
  title={RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data},
  author={Xu, Maxwell A and Narain, Jaya and Darnell, Gregory and Hallgrimsson, Haraldur and Jeong, Hyewon and Forde, Darren and Fineman, Richard and Raghuram, Karthik J and Rehg, James M and Ren, Shirley},
  journal={arXiv preprint arXiv:2411.18822},
  year={2024}
}
