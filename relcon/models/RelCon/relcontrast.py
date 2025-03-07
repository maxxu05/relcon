
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

from models.Base_Model import Base_ExpConfig

class relcontrast_ModelConfig(Base_ExpConfig):
    def __init__(self, 
                 tau=.1,
                 encoder_dims = ...,
                 withinuser_samehour_cands=1, 
                 rebardist_expconfig= None,
                 **kwargs):
        super().__init__(model_folder = "RelConContrast", **kwargs)

        
        self.tau = tau
        self.encoder_dims = encoder_dims
        self.withinuser_samehour_cands = withinuser_samehour_cands

        self.rebardist_expconfig = rebardist_expconfig


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
        from experiments.configs.crossmodalrebardist_expconfigs import allcrossmodalrebardist_expconfigs
        try:
            rebar_dist_config = allrebardist_expconfigs[rebar_dist_configname]
        except:
            rebar_dist_config = allcrossmodalrebardist_expconfigs[rebar_dist_configname]

        rebar_dist_config.set_rundir(rebar_dist_configname)

        self.rebar_dist = import_model(model_config = rebar_dist_config, 
                                       reload_ckpt=True)
        self.rebar_dist.net = self.rebar_dist.net.cuda()

    
    def setup_dataloader(self, X, y, train: bool) -> torch.utils.data.DataLoader:
        dataset = RelCon_ValidCandFolders_OnTheFly_FolderNpyDataset(X,
                                                                    data_normalizer = self.data_normalizer,
                                                                    data_clipping=self.data_clipping,
                                                                    ##############################
                                                                    withinuser_samehour_cands = \
                                                                        self.config.withinuser_samehour_cands,
                                                                    timelen = self.config.timelen
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
                # import pdb; pdb.set_trace()

                bs, candsetsize, length, channels = cand_signals.shape
                distances = []
                for idx in range(1, bs):
                    rotated_anchor_signal = torch.cat((anchor_signal[idx:, :], anchor_signal[:idx, :]), dim = 0) # compare anchor_i with anchor_(i+idx)
                    distance = self.rebar_dist.calc_distance(anchor=anchor_signal.cuda(), 
                                                            candidate=rotated_anchor_signal.cuda())
                    distances.append(distance)
                    
                for idx in range(cand_signals.shape[1]):
                    distance = self.rebar_dist.calc_distance(anchor=anchor_signal.cuda(), 
                                                             candidate=cand_signals[:,idx,:].cuda())
                    distances.append(distance)

                distances = torch.stack(distances) # shape [(bs-1)+candset_sizes, batch_size]
                # labels = torch.argmin(distances, dim=0) # for a thing in the batch, we want best of cands, so should be length bs
                # sort candidate set base on distances
                _, sortedinds = torch.sort(distances, dim = 0) # ascending order, distances increasing. 

 
                # this should be a BS x Channel output
                out_ancs = self.net(anchor_signal[:,:,self.config.encoder_dims].transpose(1,2).cuda())
                out_cands = self.net(cand_signals[:,:,:,self.config.encoder_dims].view(bs*candsetsize, length, 
                                                                                       -1).transpose(1,2).cuda())
                out_cands = out_cands.view(bs, candsetsize, -1)

                loss = relative_contrastive_loss(out_ancs, out_cands, sortedinds = sortedinds, tau=self.config.tau)

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                total_loss += loss.item() 
                # import pdb; pdb.set_trace()


            return total_loss, {}


def relative_contrastive_loss(anchor, cands, sortedinds, tau=1):
    # sortedinds bels is length (BS-1)+Candsetsize x BS, bc it tells us ordering of the cand samples
    # sorted is ascending, such that most positive (smallest dist) is first
    # anchor shape [BS, channels]
    # cands shape [BS, (BS-1)+candset_size, channels]
    bs, candset_size, channels = cands.shape

    loss = torch.zeros(bs, device=anchor.device)
    for batch_idx in range(bs):
        anchor_idx = anchor[batch_idx, :].contiguous().view(1, -1) # shape [1, channels]

        # obtain candidate set of anchor, and sort according to their distances to anchor
        # following similar rotation of anchor
        cands_idx = torch.cat((anchor[batch_idx+1:, :], anchor[:batch_idx, :], cands[batch_idx, :]), dim = 0)
        # import pdb; pdb.set_trace()
        cands_idx_revsorted = cands_idx[torch.flip(sortedinds[:, batch_idx], dims=(0,))] # shape [(BS-1)+candset_size, channels]

        sim_revsorted = F.cosine_similarity(anchor_idx, cands_idx_revsorted, dim=1)/tau
        exp_sim_revsorted = torch.exp(sim_revsorted)
        # softmax = e^(matrix - logaddexp(matrix)) = E^matrix / sumexp(matrix)
        # https://feedly.com/engineering/posts/tricks-of-the-trade-logsumexp
        cumsum_exp_sim_revsorted = torch.cumsum(exp_sim_revsorted, dim=0)

        # import pdb; pdb.set_trace() 
        # check that last cumsum is equal to last noncumsum 
        logsoftmax = sim_revsorted[1:] - torch.log(cumsum_exp_sim_revsorted[1:])
        loss_idx = -logsoftmax

        # import pdb; pdb.set_trace()
        # assert loss_idx[-1] == -F.log_softmax(sim_revsorted, dim=-1)[-1]

        loss[batch_idx] = torch.sum(loss_idx)

    return torch.mean(loss)

#################################################
import pathlib
from models.utils.CustomDatasets import OnTheFly_FolderNpyDataset, find_folders_with_python_files_recursive, filter_files_by_npy_count
import time
class RelCon_ValidCandFolders_OnTheFly_FolderNpyDataset(OnTheFly_FolderNpyDataset):
    def __init__(self, path, 
                 data_clipping=None,
                 data_normalizer=None,
                 withinuser_samehour_cands=5,
                 timelen = None):
        'Initialization'
        super().__init__(path, data_clipping=data_clipping, data_normalizer=data_normalizer, timelen=timelen)
        # import pdb; pdb.set_trace()
        # starttime = time.time()
        self.filelist = filter_files_by_npy_count(self.filelist, withinuser_samehour_cands+1)
        # endtime = time.time()
        # print(starttime-endtime)
        # self.filelist = find_folders_with_python_files_recursive(path, withinuser_samehour_cands+1)
        # endedntime = time.time()
        # print(endtime-endedntime)
        # import pdb; pdb.set_trace()
        self.length = len(self.filelist)

        self.withinuser_samehour_cands = withinuser_samehour_cands
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        superoutput_dict = super().__getitem__(idx)
        signal, filepath = superoutput_dict["signal"], superoutput_dict["filepath"]
        # print(signal.shape)
        # import pdb; pdb.set_trace()

        hourfolder = filepath.parents[0]
        signals_samehour = set(pathlib.Path(hourfolder).rglob('*.npy'))
        signals_samehour.remove(filepath)
        withinuser_samehour_cand_names= np.random.choice(list(signals_samehour), 
                                                            size=self.withinuser_samehour_cands, 
                                                            replace=False)
        withinuser_samehour_cand_signals = []
        for name in withinuser_samehour_cand_names:
            withinuser_samehour_cand_signals.append(np.load(name).astype(np.float32))

        withinuser_samehour_cand_signals = np.stack(withinuser_samehour_cand_signals)

        return signal, withinuser_samehour_cand_signals