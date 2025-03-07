from relcon.models.Base_Models import Base_ModelConfig
from relcon.nets.Base_Nets import Base_NetConfig
# from eval.Base_Eval import Base_EvalConfig
from relcon.data.Base_Datasets import SSLDataConfig, SupervisedDataConfig

from relcon.models.MotifDist.MotifDist import MotifDist_ModelConfig


allmotifdist_expconfigs = {}


allmotifdist_expconfigs["motifdist"] = MotifDist_ModelConfig(

        data_config=SSLDataConfig(
            data_folder="/home/maxxu/projects/harmfulstressors/data/ppg_acc_np_100days/",
        ),
        net_config = Base_NetConfig(net_folder="CrossAttn",
                                    net_file="CrossAttn",
                                    params = {"revin": False,
                                              "query_dimnum": 3,
                                              "key_dimnum": 3,       
                                              "kernel_size":15,
                                              "embed_dim": 64,
                                              "double_receptivefield": 2}),
        epochs = 20, lr=0.001, batch_size=64, save_epochfreq=10)