__author__ = "Sereda"
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

BWR_PATH = "C:\\!mywork\\datasets\\BWR_data_schiller\\"
ALL_LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
SELECTED_LEADS = ['i', 'ii', 'iii']
SIGNAL_LEN = 5000

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal = np.array(sample)
        return torch.from_numpy(signal).float()

class ECGDataset(Dataset):
    """ECG patches dataset."""

    def __init__(self, json_file, patch_len, transform=ToTensor()):
        """
        Args:
            json_file (string): Path to the json file with ECGs.
            patch_len (int): Number of measurements in ECG fragment
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.patch_len = patch_len
        self.indexes = list(self.data.keys())

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        ecg_object = self.data[self.indexes[idx]]
        start_of_patch = np.random.randint(low=0, high=SIGNAL_LEN-self.patch_len-1)
        signal = self.cut_patch(ecg_object, start_of_patch)
        if self.transform:
            res = self.transform(signal)
        return res

    def cut_patch(self, ecg_obj, start_of_patch):
        patch = []
        leads = ecg_obj['Leads']
        for lead_name in SELECTED_LEADS:
            lead_signal = leads[lead_name]['Signal']
            lead_patch = lead_signal[start_of_patch : start_of_patch + self.patch_len]
            patch.append(lead_patch)
        return patch



