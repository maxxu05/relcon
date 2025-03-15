import torch
import pathlib
import numpy as np

##### Dataset Configs #####
class Base_DatasetConfig:
    def __init__(
        self,
        data_folder: str,
    ):
        self.data_folder = data_folder
        self.type = None

class CV_SupervisedDataConfig(Base_DatasetConfig):
    def __init__(self, X_annotates, y_annotate: str, **kwargs):
        super().__init__(**kwargs)

        self.X_annotates = X_annotates  # ["ppg"]
        self.y_annotate = y_annotate  # "harmstress_notstress"

        self.type = "cv_supervised"

class SupervisedDataConfig(Base_DatasetConfig):
    def __init__(self, X_annotates, y_annotate: str, **kwargs):
        super().__init__(**kwargs)

        self.X_annotates = X_annotates  # ["ppg"]
        self.y_annotate = y_annotate  # "harmstress_notstress"

        self.type = "supervised"


class Npy_SupervisedDataConfig(Base_DatasetConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "npy_supervised"


class SSLDataConfig(Base_DatasetConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "ssl"


##### Dataset Classes #####
class Base_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super.__init__()


class OnTheFly_FolderNpyDataset(Base_Dataset):
    def __init__(self, path: list):
        "Initialization"
        self.path = path
        self.filelist = list(pathlib.Path(path).rglob("*.npy"))
        self.length = len(self.filelist)

    def __len__(self):
        "Denotes the total number of samples"
        return self.length

    def __getitem__(self, idx):
        "Generates one sample of data"
        signal = np.load(self.filelist[idx]).astype(np.float32).copy()
        filepath = self.filelist[idx]

        output_dict = {"signal": signal, "filepath": str(filepath)}

        return output_dict
