import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import math
from itertools import islice
import collections
import pandas as pd
import pickle
from ..dataset import PTFilesDataset

from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	tabular = torch.cat([item[1] for item in batch], dim = 0)
	label = torch.LongTensor([item[2] for item in batch])
	return [img, tabular, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 0} if device.type == "cuda" else {'num_workers': 0}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				print(split_dataset.__len__()	)
				
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)





def set_seeds(seed_value=42):
    """Set seeds for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        


def save_results_to_pkl(test_data, predict_proba, predicted, output_path):
    results_dict = {slide_id: {'slide_id': slide_id, 'prob': predict_proba[idx], 'label': predicted[idx]}
                    for idx, slide_id in enumerate(test_data['slide_id'].values)}

    with open(output_path, 'wb') as file:
        pickle.dump(results_dict, file)


def create_datasets_for_fold(split_file, pt_directory, mre_directory, label_csv, tabular_csv, mre_endo_csv):
    splits = pd.read_csv(split_file)
    train_ids = splits['train'].dropna().astype(int).tolist()
    val_ids = splits['val'].dropna().astype(int).tolist()
    test_ids = splits['test'].dropna().astype(int).tolist()

    train_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, train_ids, mre_endo_csv)
    val_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, val_ids, mre_endo_csv)
    test_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, test_ids, mre_endo_csv)

    return train_dataset, val_dataset, test_dataset


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        endo_data, mre_data, tabular_data, labels = batch
        mre_data, endo_data, tabular_data, labels = [data.to(device) for data in (mre_data, endo_data, tabular_data, labels)]
        labels_one_hot = F.one_hot(labels, num_classes=2).float()

        logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
        loss = criterion(logits, labels_one_hot)
        instance_loss_endo = results_dict_endo['instance_loss']
        instance_loss_mre = results_dict_mre['instance_loss']

        total_loss = 1.7 * loss + instance_loss_endo + instance_loss_mre
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            endo_data, mre_data, tabular_data, labels = batch
            mre_data, endo_data, tabular_data, labels = [data.to(device) for data in (mre_data, endo_data, tabular_data, labels)]
            labels_one_hot = F.one_hot(labels, num_classes=2).float()

            logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
            loss = criterion(logits, labels_one_hot)
            total_loss += 1.7 * loss + results_dict_endo['instance_loss'] + results_dict_mre['instance_loss']

    return total_loss / len(val_loader)


def test(model, test_loader, fold, split_file, results_dir, device):
    model.eval()
    all_labels, all_predictions, Y_prob = [], [], []
    test_ids = pd.read_csv(split_file)['test'].dropna().astype(int).tolist()

    with torch.no_grad():
        for batch in test_loader:
            endo_data, mre_data, tabular_data, labels = batch
            mre_data, endo_data, tabular_data, labels = [data.to(device) for data in (mre_data, endo_data, tabular_data, labels)]

            logits, _, _ = model(endo_data, mre_data, tabular_data, label=labels)
            probabilities = softmax(logits, dim=1)
            Y_prob.extend(probabilities.cpu().numpy())
            _, predicted = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    predict_proba = [prob[1] for prob in Y_prob]
    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss