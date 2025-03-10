from relcon.nets.Base_Nets import Base_NetConfig

# from eval.Base_Eval import Base_EvalConfig
from relcon.data.Base_Dataset import SSLDataConfig, SupervisedDataConfig

from relcon.models.RelCon.RelCon_Model import RelCon_ModelConfig


allrelcon_expconfigs = {}


allrelcon_expconfigs["25_3_8_relcon"] = RelCon_ModelConfig(
    withinuser_cands=5,
    motifdist_expconfig_key="25_3_8_motifdist",
    data_config=SSLDataConfig(
        data_folder="relcon/data/datasets/dummydataset",
    ),
    net_config=Base_NetConfig(
        net_folder="ResNet1D",
        net_file="ResNet1D_Net",
        params={
            "init_conv": {
                "in_channels": 3,
                "base_filters": 64,
                "kernel_size": 7,
                "stride": 2,
                "padding": 0,
            },
            "init_maxpool": {
                "kernel_size": 3,
                "stride": 2,
            },
            "blocks": [
                {
                    "n": 3,
                    "in_channels": 64,
                    "base_filters": 64,
                    "kernel_size": 3,
                    "stride": 1,
                },
                {
                    "n": 4,
                    "in_channels": 64,
                    "base_filters": 128,
                    "kernel_size": 4,
                    "stride": 1,
                },
                {
                    "n": 6,
                    "in_channels": 128,
                    "base_filters": 256,
                    "kernel_size": 4,
                    "stride": 1,
                },
            ],
            "finalpool": "avg",
        },
    ),
    epochs=100,
    lr=0.000001,
    batch_size=64,
    save_epochfreq=10,
)
