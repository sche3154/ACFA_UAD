from data.io.base_io import BaseIO
from utils.dmri_io import load_data, save_data, load_random6DWIs
from data.io.QC_sample import QCSample
import os 
import numpy as np

class HCPUADIO(BaseIO):
    
    def __init__(self, opt):
        BaseIO.__init__(self, opt)

    def load_sample(self, index):

        if self.opt.isTrain:
            b0_dwis_path = os.path.join(self.root, 'single_shell', index+'_DWI_processed_b1000.mif.gz')
            b0, dwis, affine = load_random6DWIs(b0_dwis_path, needs_affine= True)

            dti_path = os.path.join(self.root, 'dti_nii_tensor_only', index+'_DTI.nii.gz')
            dti = load_data(dti_path)

            fa_path = os.path.join('/home/sheng/data', 'FA',  index+'_dti_FA.nii.gz')
            fa = load_data(fa_path)

            bm_path = os.path.join(self.root, 'mask/mask_RAS', index + '_DWI_brainmask.mif.gz')
            brain_mask = load_data(bm_path)
        else:
            eval_path = '/home/sheng/data/eval_data'
            # print(self.root)
            b0_dwis_path = os.path.join(eval_path
                                        , index, index+'_DWI_6dir_eval.nii.gz')
            # print(b0_dwis_path)
            b0_dwis, affine = load_data(b0_dwis_path, needs_affine= True)
            # print(b0_dwis.shape)
            b0, dwis = b0_dwis[...,0], b0_dwis[...,1:]

            bm_path = os.path.join(eval_path , index, index + '_DWI_brainmask_eval.mif.gz')
            brain_mask = load_data(bm_path)

            label_path = os.path.join(eval_path, index, index + '_label.npy')
            label = np.load(label_path)

            dti_path = os.path.join(eval_path, index,index+'_DTI_eval.nii.gz')
            dti = load_data(dti_path)

            fa_path = os.path.join(eval_path, index, index+'_FA_eval.nii.gz')
            fa = load_data(fa_path)

        sample = QCSample(index, b0, dwis, affine)
        sample.brain_mask = brain_mask
        sample.dti = dti
        sample.fa = fa

        if self.opt.isTrain is False:
            sample.label = label
        
        return sample
    
    def save_sample(self, sample):
        value, affine, index, name = sample
        output_name = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}.nii.gz'.format(index, name))
        save_data(value, affine=affine, output_name=output_name)

        

    