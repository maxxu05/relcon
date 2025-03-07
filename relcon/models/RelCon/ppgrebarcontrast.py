
from models.Base_Model import BaseModelClass
from experiments.configs.rebarcontrast_expconfigs import rebarcontrast_ExpConfig
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from sklearn import metrics
from tqdm import tqdm
from utils.utils import printlog, load_data, import_model, init_dl_program


class Model(BaseModelClass):
    def __init__(
        self,
        *args,
        **kwargs
        ):
        super().__init__(*args,**kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

        rebar_dist_configname = self.config.rebardist_expconfig
        from experiments.configs.rebardist_expconfigs import allrebardist_expconfigs
        rebar_dist_config = allrebardist_expconfigs[rebar_dist_configname]
        rebar_dist_config.set_rundir(rebar_dist_configname)

        self.rebar_dist = import_model(model_config = rebar_dist_config, 
                                       reload_ckpt=True)
        self.rebar_dist.net = self.rebar_dist.net.cuda()

    
    def setup_dataloader(self, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = RelCon_ValidCandFolders_OnTheFly_FolderNpyDataset(X,
                                                                    withinuser_samehour_cands = \
                                                                        self.config.withinuser_samehour_cands
                                                                    )
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers= 0)# torch.get_num_threads())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers= torch.get_num_threads())

        return loader
    
    def create_state_dict(self, epoch: int, test_loss) -> dict:
        state_dict = {"net": self.net.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "test_loss": test_loss,
                      "epoch": epoch}

        return state_dict

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.net.train(mode = train)
        self.optimizer.zero_grad()

        # save outputs and attention for visualization
        # if not train:
        #     reconstruction_list, attn_list, mask_list, x_original_list = [], [], [], []
        with torch.set_grad_enabled(train):
            total_loss = 0 

            for out_tuple in tqdm(dataloader, desc="Training" if train else "Evaluating", leave=False):
                anchor_signal = out_tuple[0]
                cand_signals = out_tuple[1] # .view(-1, anchor_signal.shape[1], anchor_signal.shape[2])
                
                distances = []
                for idx in range(cand_signals.shape[1]):
                    distance = self.rebar_dist.calc_distance(anchor=anchor_signal.cuda(), 
                                                             candidate=cand_signals[:,idx,:].cuda())
                    distances.append(distance)

                distances = torch.stack(distances) # shape [candset_sizes, batch_size]
                labels = torch.argmin(distances, dim=0) # for a thing in the batch, we want best of cands, so should be length bs

                out_ancs = self.net(anchor_signal.transpose(1,2).cuda())
                out_cands = self.net(cand_signals.view(-1, anchor_signal.shape[1], 
                                                       anchor_signal.shape[2]).transpose(1,2).cuda())

                loss = contrastive_loss_imp(z1= out_ancs, z2= out_cands, labels = labels,
                                            alpha=self.config.alpha, tau=self.config.tau)

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                total_loss += loss.item() 


            return total_loss, {}



def find_folders_with_python_files_recursive(directory, min_py_files=5):
    """
    Recursively search through folders and find those containing at least a certain number of .py files.

    :param directory: Root directory to scan.
    :param min_py_files: Minimum number of .py files to qualify.
    :return: List of folder paths meeting the criteria.
    """
    qualifying_folders = []

    # Walk through each directory in the tree
    for dir in [x[0] for x in os.walk(directory)]:
        # Check each immediate subdirectory of the current root
        dirpath = os.path.join(directory, dir)

        
        # Count .py files in the immediate level of this subdirectory
        py_file_count = sum(
            1 for file in os.listdir(dirpath)
            if file.endswith('.npy') and os.path.isfile(os.path.join(dirpath, file))
        )

        # Add subdirectory to results if it meets the criteria
        if py_file_count >= min_py_files:
            qualifying_folders.extend([pathlib.Path(os.path.join(dirpath, file)) for file in os.listdir(dirpath)])

    return qualifying_folders


import pathlib
class RelCon_ValidCandFolders_OnTheFly_FolderNpyDataset(torch.utils.data.Dataset):
    def __init__(self, path: list, withinuser_samehour_cands=5,):
        'Initialization'
        self.path = path
        # self.filelist = list(pathlib.Path(path).rglob('*.npy'))
        self.filelist = find_folders_with_python_files_recursive(path, withinuser_samehour_cands+1)
        self.length = len(self.filelist)

        self.withinuser_samehour_cands = withinuser_samehour_cands

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, idx):
        'Generates one sample of data'
        filepath = self.filelist[idx]
        signal = np.load(filepath).astype(np.float32)[:, np.newaxis,]

        hourfolder = filepath.parents[0]
        signals_samehour = set(pathlib.Path(hourfolder).rglob('*.npy'))
        signals_samehour.remove(filepath)
        withinuser_samehour_cand_names= np.random.choice(list(signals_samehour), 
                                                            size=self.withinuser_samehour_cands, 
                                                            replace=False)
        withinuser_samehour_cand_signals = []
        for name in withinuser_samehour_cand_names:
            withinuser_samehour_cand_signals.append(np.load(name).astype(np.float32))

        withinuser_samehour_cand_signals = np.stack(withinuser_samehour_cand_signals)[:, :, np.newaxis,]

        # output_dict = {"anchor_signal": signal,
        #                "withinuser_samehour_cand_signals": withinuser_samehour_cand_signals}
        return signal, withinuser_samehour_cand_signals
    

def contrastive_loss_imp(z1, z2, labels, tau=1, alpha=0.5):
    # labels is length BS, bc it tells us which of the cand samples is the best
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]
    z1 = F.max_pool1d(
        z1.transpose(1, 2).contiguous(),
        kernel_size = z1.size(1)).transpose(1, 2)
    z2 = F.max_pool1d(
        z2.transpose(1, 2).contiguous(),
        kernel_size = z2.size(1)).transpose(1, 2)

    loss =  instance_contrastive_loss_imp(z1, z2, labels, tau=tau) 
    loss *= alpha 
    loss += (1 - alpha) * temporal_contrastive_loss_imp(z1, z2, labels, tau=tau)
    return loss.to(device=z1.device)
    

def instance_contrastive_loss_imp(z1, z2, labels, tau=1): 
    # for a given time, other stuff in the batch is negative
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]

    # need to get this T x 2B x 2B
    bs, ts_len, channels, = z1.shape
    candset_size = z2.shape[0]//bs

    loss = torch.zeros(bs, device=z1.device)
    for batch_idx in range(bs):
        # [1 x channel] x [channel x candset_size]
        # I want a 1 x candset_size 
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        # for batch_idx 3, we know the 4th mc is the best, so to get there. we go to the 4th mc by doing 4*bs and then going to the + batch idx
        positive = z2[labels[batch_idx]*bs+batch_idx, :].contiguous().view(1, -1)
        negatives = torch.cat((z1[:batch_idx, :].contiguous().view(-1, positive.shape[-1]), z1[batch_idx+1:, :].contiguous().view(-1, positive.shape[-1])))
        temp_z2 = torch.cat((positive, negatives))
        
        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)
        logits = -F.log_softmax(sim/tau, dim=-1)
        loss[batch_idx] = logits[0,0]

    return loss.mean()


def temporal_contrastive_loss_imp(z1, z2, labels, tau=1):
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]
    bs, ts_len, channels, = z1.shape
    candset_size = z2.shape[0]//bs

    loss = torch.zeros(bs, device=z1.device)
    for batch_idx in range(bs):
        # with time as a dimension so you could do it by, this means first time must be the same as the other time step
        # [1 x length*channel] x [length*channel x candset_size]
        # better way could be cosine similarity with a set
        # [1 x channel] x [channel x candset_size]
        # I want a 1 x candset_size 
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        temp_z2 = z2[batch_idx::bs, :].contiguous().view(candset_size, -1) # positive is batch_idx + bs*(labels[batch_idx])
        
        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)
        logits = -F.log_softmax(sim/tau, dim=-1)
        loss[batch_idx] = logits[0, labels[batch_idx]] 

    return loss.mean()

#################################################


class RelCon_OnTheFly_FolderNpyDataset(torch.utils.data.Dataset):
    def __init__(self, path: list, withinuser_samehour_cands=5,):
        'Initialization'
        self.path = path
        self.filelist = list(pathlib.Path(path).rglob('*.npy'))
        self.length = len(self.filelist)

        self.withinuser_samehour_cands = withinuser_samehour_cands


    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, idx):
        'Generates one sample of data'
        filepath = self.filelist[idx]
        signal = np.load(filepath).astype(np.float32)

        hourfolder = filepath.parents[0]
        signals_samehour = set(pathlib.Path(hourfolder).rglob('*.npy'))
        signals_samehour.remove("filepath")
        withinuser_samehour_cand_signals = np.random.choice(signals_samehour, 
                                                            size=self.withinuser_samehour_cands, 
                                                            replace=False)

        output_dict = {"anchor_signal": signal,
                       "withinuser_samehour_cand_signals": withinuser_samehour_cand_signals}
        return output_dict