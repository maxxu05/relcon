"""
Code needed to reconstruct data splits and evaluations used to compare against Yuan et al 2022's prior 
large-scale pre-trained model. This was used to construct the results from Table 3 in the paper, copy 
and pasted below. 

+-------------------------------------------------------------------------------------------------------------+
|                    Table 3: RelCon FM compared to Prior Large-Scale Pre-trained Model.                      |
+-----------------+---------------+--------------+--------------+---------------+--------------+--------------+
|                 | Eval Method   | Pre-train    | Opportunity (Wrist→Wrist)    | PAMAP2 (Wrist→Wrist)        |
|                 |               | Data         | F1 ↑         | Kappa ↑       | F1 ↑         | Kappa ↑      |
+-----------------+---------------+--------------+--------------+---------------+--------------+--------------+
| RelCon FM       | MLP Probe     | AHMS         | 69.1 ± 8.3   | 62.1 ± 6.2    | 85.4 ± 3.6   | 84.7 ± 3.6   |
| Yuan et al. FM  | MLP Probe     | UKBioBank    | 57.0 ± 7.8   | 43.5 ± 9.2    | 72.5 ± 5.4   | 71.7 ± 5.7   |
+-----------------+---------------+--------------+--------------+---------------+--------------+--------------+
| RelCon FM       | Fine-tuned    | AHMS         | 98.4 ± 0.9   | 97.9 ± 0.8    | 98.8 ± 1.3   | 98.6 ± 1.6   |
| Yuan et al. FM  | Fine-tuned    | UKBioBank    | 59.5 ± 8.5   | 47.1 ± 10.4   | 78.9 ± 5.4   | 76.9 ± 5.9   |
+-----------------+---------------+--------------+--------------+---------------+--------------+--------------+
| RelCon FM       | From Scratch  | n/a          | 94.0 ± 1.3   | 92.8 ± 1.8    | 97.5 ± 1.1   | 97.0 ± 1.3   |
| Yuan et al. FM  | From Scratch  | n/a          | 38.3 ± 12.4  | 23.8 ± 15.4   | 60.5 ± 8.6   | 59.6 ± 8.6   |
+-------------------------------------------------------------------------------+--------------+--------------+


"""

import numpy as np
import os
from tqdm import tqdm
from relcon.data.process.comparepriorpt.oppo import main as main_oppo
from relcon.data.process.comparepriorpt.pamap import main as main_pamap

RAW_DIRECTORY = "relcon/data/datasets/priorpt/"

def main():
    os.makedirs(RAW_DIRECTORY, exist_ok=True)
    main_oppo(RAW_DIRECTORY)
    main_pamap(RAW_DIRECTORY)


if __name__ == "__main__":
    main()

