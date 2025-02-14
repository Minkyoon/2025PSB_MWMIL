from torch.utils.data import Dataset,  DataLoader, WeightedRandomSampler
import pandas as pd
import torch
import os


#mre endo matching
def match_mre_endo(mre_endo_csv):
    mre_endo_data = pd.read_csv(mre_endo_csv)
    matched_data = {}
    for index, row in mre_endo_data.iterrows():
        endo_id = row['serial_number_endo']
        mre_id = row['serial_number_mri']
        matched_data[endo_id] = mre_id
    return matched_data



class PTFilesDataset(Dataset):
    def __init__(self, pt_directory, mre_directory, label_csv, tabular_csv, ids=None, mre_endo_csv=None):
        self.pt_directory = pt_directory
        self.labels = pd.read_csv(label_csv)
        self.labels.set_index("slide_id", inplace=True)
        self.tabular_data = pd.read_csv(tabular_csv).drop(columns=['date'])
        self.tabular_data.set_index("slide_id", inplace=True)
        self.tabular_data.fillna(0, inplace=True)
        self.ids = ids
        self.mre_directory = mre_directory
        self.slide_cls_ids = self._get_class_counts()
        if mre_endo_csv:
            self.mre_endo_match = match_mre_endo(mre_endo_csv)
        else:
            self.mre_endo_match = {}

    def __len__(self):
        if self.ids is not None:
            return len(self.ids)
        return len(os.listdir(self.pt_directory))

    def getlabel(self, idx):
        if self.ids is not None:
            slide_id = self.ids[idx]
        else:
            file_name = os.listdir(self.pt_directory)[idx]
            slide_id = int(file_name.split('.')[0])
        return self.labels.loc[slide_id, "label"]

    def _get_class_counts(self):
        class_counts = {}
        for idx in range(len(self.labels)):
            label = self.labels.iloc[idx]['label']
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        return class_counts

    def __getitem__(self, idx):
        if self.ids is not None:
            slide_id = self.ids[idx]
        else:
            file_name = os.listdir(self.pt_directory)[idx]
            slide_id = int(file_name.split('.')[0])

        label = self.labels.loc[slide_id, "label"]
        file_path = os.path.join(self.pt_directory, f"{slide_id}.pt")
        data = torch.load(file_path)
        
        if slide_id in self.mre_endo_match:
            mre_id = self.mre_endo_match[slide_id]
            mre_file_path = os.path.join(self.mre_directory, f"{mre_id}.pt")
            mre_data = torch.load(mre_file_path)

        tabular_features = self.tabular_data.loc[slide_id].values
        tabular_features = torch.tensor(tabular_features, dtype=torch.float32)

        return data, mre_data, tabular_features, label

    


def make_weights_for_balanced_classes(dataset):
    N = float(len(dataset))
    weight_per_class = [N / dataset.slide_cls_ids[c] for c in dataset.slide_cls_ids]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)

def get_split_loader(split_dataset, training=False, weighted=False):
    kwargs = {'num_workers': 4} if torch.cuda.is_available() else {}
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes(split_dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = torch.utils.data.RandomSampler(split_dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(split_dataset)

    loader = DataLoader(split_dataset, batch_size=1, sampler=sampler)
    return loader