import os
import nibabel as nib
import pickle
from utils.mrtrix import *

def load_data(path, needs_affine = False):

    if not os.path.exists(path):
        raise ValueError(
            "Data could not be found \"{}\"".format(path)
        )
        exit(0)

    if path.endswith('.mif.gz') or path.endswith('.mif'):
        vol = load_mrtrix(path)
        data_copied = vol.data.copy()
        affine_copied = vol.transform.copy()
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        vol = nib.load(path)
        data_copied = vol.get_fdata().copy()
        affine_copied = vol.affine.copy()
    else:
        raise IOError('file extension not supported: ' + str(path))
        exit(0)

    # Return volume
    if needs_affine:
        return data_copied, affine_copied
    else:
        return data_copied
    
def load_random6DWIs(path, needs_affine = False):

    if not os.path.exists(path):
        raise ValueError(
            "Data could not be found \"{}\"".format(path)
        )
        exit(0)

    if path.endswith('.mif.gz') or path.endswith('.mif'):
        vol = load_mrtrix(path)
        data_copied = vol.data.copy()
        affine_copied = vol.transform.copy()
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        vol = nib.load(path)
        data_copied = vol.get_fdata().copy()
        affine_copied = vol.affine.copy()
    else:
        raise IOError('file extension not supported: ' + str(path))
        exit(0)

    b0 = data_copied[...,0]
    random6_index = np.random.randint(1,91, size = 6)
    random6_dwis = data_copied[...,random6_index]

    return b0, random6_dwis, affine_copied

def save_data(data, affine, output_name):
    nifti = nib.Nifti1Image(data, affine=affine)
    nib.save(nifti, output_name)
    print('Save image to the path {:}'.format(output_name))

def get_HCPsamples(hcp_split_path, train =True):
    if not os.path.exists(hcp_split_path):
        raise IOError(
            "hcp splited list path, {}, could not be resolved".format(hcp_split_path)
        )
        exit(0)

    with open(hcp_split_path, 'rb') as handle:
        sub_list = pickle.load(handle)
        
        if train:
            sample_list = sub_list['train']
        else:
            sample_list = sub_list['test']
            
    return sample_list

def get_HCPEvalsamples(hcp_eval_path):
    if not os.path.exists(hcp_eval_path):
        raise IOError(
            "hcp eval splited list path, {}, could not be resolved".format(hcp_eval_path)
        )
        exit(0)

    with open(hcp_eval_path, 'rb') as handle:
        sub_list = pickle.load(handle)
        
        sample_list = []
        sample_list.extend(sub_list['normal'])
        sample_list.extend(sub_list['abnormal'])
        
    return sample_list

def save_pickle(file, path):

    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):

    with open(path, 'rb') as handle:
        pickle_file = pickle.load(handle)

    return pickle_file

def list_dir(directory):
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)
    return visible_files