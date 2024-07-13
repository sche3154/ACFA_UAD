from data.processing.base_processing import BaseProcessing
from utils.patch_operations import concat_matrices, pad_3d_image, unpad_3d_image
import numpy as np
import os
import json

class UADProcessing(BaseProcessing):

    def __init__(self, opt):
        super().__init__(opt)
        self.kernel_size = (64,64,64)
        self.stride_size = (32,32,32)
        # self.lfov = 32
    
    def preprocessing(self, sample):
        b0 = sample.b0.squeeze()
        dwis = sample.dwis.squeeze()
        bm = sample.brain_mask.squeeze()
        dti = sample.dti.squeeze()
        fa = sample.fa.squeeze()
        fa[fa>=1] = 1.0
        ## Norm 
        print('{:} the {:}'.format(self.opt.data_norm, sample.index))
        norm_method = getattr(self, self.opt.data_norm)
        b0_norm, _, _ = norm_method(b0, bm)
        dwis_norm, _, _ = norm_method(dwis, bm)

        ## Bounding 
        print('Bounding the {:}'.format(sample.index))
        x_1, x_2, y_1, y_2, z_1, z_2 = self.find_bounding_box_3D(bm)
        bb = [x_1, x_2, y_1, y_2, z_1, z_2]
        b0_bounded = b0_norm[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        dwis_bounded = dwis_norm[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        bm_bounded = bm[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        dti_bounded = dti[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        fa_bounded = fa[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        # print(b0_bounded.shape)

        if self.opt.input_patch_size > 0:
            print('Patching the {:}'.format(sample.index))
            b0_patches, coords = self.patch_generation(b0_bounded
                                                , self.kernel_size, self.stride_size, three_dim=True, index=sample.index)
            dwi_patches = self.patch_generation_by_coords(dwis_bounded, coords= coords)
            dti_patches = self.patch_generation_by_coords(dti_bounded, coords=coords)
            fa_patches = self.patch_generation_by_coords(fa_bounded, coords=coords)
            bm_patches = self.patch_generation_by_coords(bm_bounded, coords= coords)

            ## Updating 
            sample.bb = bb
            sample.bb_shape = b0_bounded.shape
            sample.b0_processed = b0_patches
            sample.dwis_processed = dwi_patches
            sample.dti_processed = dti_patches
            sample.fa_processed = fa_patches
            sample.bm_processed = bm_patches
        else:
            ## Padding 
            print('Padding the {:}'.format(sample.index))
            b0_padded = pad_3d_image(b0_bounded, (160,192,160), padding_value=0)
            # print(dwis_bounded.shape)
            dwis_padded = pad_3d_image(dwis_bounded, (160,192,160,6), padding_value=0)
            bm_padded = pad_3d_image(bm_bounded, (160,192,160), padding_value=0)
            dti_padded = pad_3d_image(dti_bounded, (160,192,160,6), padding_value=0)
            fa_padded = pad_3d_image(fa_bounded, (160,192,160,6), padding_value=0)
            # print(dwis_padded.shape)
            sample.bb = bb
            sample.bb_shape = b0_bounded.shape
            sample.b0_processed = b0_padded
            sample.dwis_processed = dwis_padded
            sample.dti_processed = dti_padded
            sample.fa_processed = fa_padded
            sample.bm_processed = bm_padded

    def postprocessing(self, preprocessed_sample):
        target =  preprocessed_sample
        out_dict = preprocessed_sample.out_dict
        
        resulted_dict = {}
        for key, value in out_dict.items(): # （patches, value)

            # print(value.shape)
            if  'pred' in key or "inputs" in key:
                # print(value.shape)
                resulted_img = value
                # print(value.shape)
                if self.opt.input_patch_size > 0:
                    print('Un-patching the {:} of {:}'.format(key, target.index))
                    # print(value.shape)
                    w,h,d,c = target.bb_shape[0], target.bb_shape[1], target.bb_shape[2], value.shape[-1]
                    resulted_img = concat_matrices(patches=value,
                                                image_size = (w,h,d,c),
                                                window= self.kernel_size,
                                                overlap= self.stride_size,
                                                three_dim= True,
                                                coords=target.coords)
                else:
                    print('Un-padding the {:} of {:}'.format(key, target.index))
                    w,h,d = target.bb_shape
                    resulted_img = unpad_3d_image(value, original_shape=(w,h,d,value.shape[-1]))
                    # print(resulted_img.shape)

                print('Un-bounding the {:} of {:}'.format(key, target.index))
                w,h,d,c = target.shape[0], target.shape[1], target.shape[2], value.shape[-1]
                tmp = np.zeros((w,h,d,c))
                bb = target.bb
                tmp[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = resulted_img

                resulted_img = tmp
                resulted_dict[key] = resulted_img
                                 
        return resulted_dict
