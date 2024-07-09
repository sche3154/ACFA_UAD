from data.base_dataset import BaseDataset
from data.io import create_io
from data.processing import create_prcoessing
from utils.dmri_io import get_HCPsamples, get_HCPEvalsamples
import os
import torch
import numpy as np
 
class HCPUADDataset(BaseDataset):  

    def __init__(self, opt):
        BaseDataset.__init__(self,opt)

        if self.opt.isTrain:
            hcp_split_path = os.path.join(self.root, 'HCP_list_split_80_20.pickle')
            self.sample_list = get_HCPsamples(hcp_split_path, train = self.opt.isTrain)
        else:
            hcp_split_path = os.path.join('/home/sheng/data'
            , 'HCP_eval_list.pickle')
            self.sample_list = get_HCPEvalsamples(hcp_split_path)

        self.io = create_io('hcpUAD', opt)
        self.processing = create_prcoessing('UAD', opt)

        # self.sample_list = ['159340', '211720', '147737', '672756', '103818', '100307']

    def __len__(self):
        """Return the total number of images in the dataset."""

        return len(self.sample_list)
    
    def __getitem__(self, index):

        sample_index = self.sample_list[index]
        sample = self.io.load_sample(sample_index)
        self.processing.preprocessing(sample)

        if self.opt.isTrain is False:
            self.preprocessed_sample = sample

        b0 = torch.from_numpy(sample.b0_processed).unsqueeze(0) # (1,w,h,d)
        dwis = torch.from_numpy(sample.dwis_processed).permute(3,0,1,2)  # (c,w,h,d)
        bm = torch.from_numpy(sample.bm_processed).unsqueeze(0) # (1,w,h,d)

        dti = torch.from_numpy(sample.dti_processed).permute(3,0,1,2)  # (c,w,h,d)
        fa = torch.from_numpy(sample.fa_processed).unsqueeze(0)  # (c,w,h,d)
        
        items = {'b0': b0
                ,'dwis': dwis, 'dti': dti, 'fa': fa
                ,'bm': bm}
        
        return items
    
    def postprocessing(self, outputs, counter):
        #  outputs: dicts of multi output
        self.preprocessed_sample.out_dict = outputs
        resulted_dict = self.processing.postprocessing(self.preprocessed_sample)

        if counter % self.opt.save_prediction == 0:
            for key in resulted_dict.keys():
                name = key
                self.io.save_sample((resulted_dict[key], self.preprocessed_sample.affine
                                          , self.preprocessed_sample.index, name))
