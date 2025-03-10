"""
Code needed to reconstruct data splits and evaluations used to compare against Haresamudram et al 2022's  
accelerometry SSL benchmarking study. This was used to construct the results from Table 4 in the paper, copy 
and pasted below. 
+----------------------------------------------------------------------------------------------+
|                Table 4: RelCon FM compared to an Accel SSL Benchmarking Study                |
+-------------------------+------------------+-----------------+----------------+--------------+
|                         |                  | HHAR            | Motionsense    | PAMAP2       |
|                         |                  | (Wrist→Wrist)   | (Wrist→Waist)  | (Wrist→Leg)  |
|                         |                  | F1 ↑            | F1 ↑           | F1 ↑         |
+-------------------------+------------------+-----------------+----------------+--------------+
| Self-supervised w/      | RelCon FM        | 57.63 ± 3.24    | 80.35 ± 0.71   | 53.98 ± 0.76 |
| Frozen Embedding +      +------------------+-----------------+----------------+--------------+
| Linear Probe            | Aug Pred         | 50.95 ± 2.70    | 74.96 ± 1.37   | 46.90 ± 1.14 |
|                         | SimCLR           | 55.93 ± 1.75    | 83.93 ± 1.78   | 50.75 ± 2.97 |
|                         | SimSiam          | 45.36 ± 4.98    | 71.91 ± 12.30  | 47.85 ± 2.48 |
|                         | BYOL             | 40.66 ± 4.08    | 66.44 ± 2.76   | 43.89 ± 3.35 |
|                         | MAE              | 43.48 ± 2.84    | 61.14 ± 3.45   | 42.32 ± 1.63 |
|                         | CPC              | 56.24 ± 0.98    | 72.89 ± 2.06   | 45.84 ± 1.39 |
|                         | Autoencoder      | 53.57 ± 1.14    | 55.12 ± 3.46   | 50.79 ± 1.09 |
+-------------------------+------------------+-----------------+--------------+----------------+
| Fully Supervised        | DeepConvLSTM     | 54.39 ± 2.28    | 84.56 ± 0.85   | 51.22 ± 1.91 |
|                         | Conv classifier  | 55.43 ± 1.21    | 89.25 ± 0.50   | 59.76 ± 1.53 |
|                         | LSTM classifier  | 37.42 ± 5.04    | 86.74 ± 0.29   | 48.61 ± 1.82 |
+-------------------------+------------------+-----------------+--------------+----------------+

"""


import numpy as np
import os
from tqdm import tqdm
from relcon.data.process.comparesslbench.hhar import main as main_hhar
from relcon.data.process.comparesslbench.motionsense import main as main_motionsense
from relcon.data.process.comparesslbench.pamap import main as main_pamap

RAW_DIRECTORY = "relcon/data/datasets/sslbench/"

def main():
    os.makedirs(RAW_DIRECTORY, exist_ok=True)
    main_hhar(RAW_DIRECTORY)
    main_motionsense(RAW_DIRECTORY)
    main_pamap(RAW_DIRECTORY)


if __name__ == "__main__":
    main()