__author__ = "Sereda"

from enum import Enum
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DC_GAN_centered_qrs.saver import save_batch_to_images

ALL_LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
SIGNAL_LEN = 5000


class CycleComponent(Enum):
    P = 1
    QRS = 2
    T = 3


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal = np.array(sample)
        return torch.from_numpy(signal).float()


class ECGDataset(Dataset):
    """ECG patches dataset."""

    def __init__(self, patch_len,
                 transform=ToTensor(),
                 selected_leads=['i', 'ii', 'iii'],
                 what_component=CycleComponent.QRS):
        """
        Args:
            what_component (CycleComponent): we center patch at the center of this type of components
            selected_leads (array of strings): few of 12 possible leads names
            patch_len (int): Number of measurements in ECG fragment
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        BWR_PATH = "C:\\!mywork\\datasets\\BWR_ecg_200_delineation\\"
        FILENAME = "ecg_data_200.json"
        json_file = BWR_PATH + FILENAME
        self.transform = transform
        self.selected_leads = selected_leads
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.patch_len = patch_len
        self.indexes = list(self.data.keys())
        self.what_component = what_component

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        ecg_object = self.data[self.indexes[idx]]
        triplets = self.get_all_comlexes(ecg_object, self.what_component)
        random_triplet_id = np.random.randint(low=0, high=len(triplets))
        center_of_component = triplets[random_triplet_id][1]
        patch_start = center_of_component - int(self.patch_len/2)
        patch_end = patch_start + self.patch_len
        if patch_start >= 0 and patch_end < SIGNAL_LEN:
            # we can  take this patch
            signal = self.cut_patch(ecg_object, patch_start)
            if self.transform:
                res = self.transform(signal)
        else:
            # we can not take this patch, need to make another attempt
            res = self.__getitem__(idx)
        return res

    def cut_patch(self, ecg_obj, start_of_patch):
        patch = []
        leads = ecg_obj['Leads']
        for lead_name in self.selected_leads:
            lead_signal = leads[lead_name]['Signal']
            lead_patch = lead_signal[start_of_patch : start_of_patch + self.patch_len]
            patch.append(lead_patch)
        return patch

    def get_all_comlexes(self, ecg_obj, cycle_component):
        leads = ecg_obj['Leads']
        some_lead = self.selected_leads[0]
        delineation_tables = leads[some_lead]['DelineationDoc']
        triplets = None
        if cycle_component == CycleComponent.P:
            triplets = delineation_tables['p']
        else:
            if cycle_component == CycleComponent.QRS:
                triplets = delineation_tables['qrs']
            else:
                if cycle_component == CycleComponent.T:
                    triplets = delineation_tables['t']
        return triplets


if __name__ == "__main__":
    # ---------------------------------------------
    # Example of use of the above ECGDataset class:
    # let's visualise some ECGs!
    # ---------------------------------------------
    os.makedirs("images", exist_ok=True)
    dataset_object = ECGDataset(patch_len=512)
    dataloader = DataLoader(dataset_object, batch_size=15, shuffle=True)

    for i, batch in enumerate(dataloader):
        save_batch_to_images("REAL_ECG", batch)

