import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
from torch.utils.data import Dataset, DataLoader

def list_file_paths(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for file in sorted(files):
            file_list.append(os.path.join(root, file))
    return file_list

train_dir = './data/Drone-Haze/train/'
train_paths = list_file_paths(os.path.join(train_dir, 'input'))

def histogram_loader(path):
    image = skimage.io.imread(path)
    R_hist, R_bins = np.histogram(image[:, :, 0], bins=256, range=(0, 256)) # R_hist.shape = (256,)
    G_hist, G_bins = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(image[:, :, 2], bins=256, range=(0, 256))
    R_pdf = R_hist/sum(R_hist)
    G_pdf = G_hist/sum(G_hist)
    B_pdf = B_hist/sum(B_hist)
    RGB = np.vstack((R_pdf,G_pdf,B_pdf))
    return RGB


class trainset(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader

        self.images = train_paths

    def __getitem__(self, index):

        single_img = self.images[index]
        single_label = self.get_gt_image_path(single_img)
        img_hist = self.histogram_loader(single_img)
        label_hist = self.histogram_loader(single_label)
        
        img_hist = torch.tensor(img_hist,dtype=torch.float)#.permute(1,0)#.unsqueeze(1) [3,256]
        label_hist = torch.tensor(label_hist,dtype=torch.float)#.permute(1,0)#.unsqueeze(1) [3,256]
        
        single_img_np = self.images[index]        
        single_label_np = self.get_gt_image_path(single_img_np)
        
        return img_hist, label_hist, single_img_np, single_label_np#ori_img, hs_img 

    def __len__(self):
        return len(self.images)

    def get_gt_image_path(self, input_image_path):
        new_path = re.sub(r'input/([^/]+)/Haze-[123]/', r'gt/\1/', input_image_path)
        new_path = new_path.replace('_synt', '')

        return new_path

class testset(Dataset):
    def __init__(self, test_dir):
        self.histogram_loader = histogram_loader

        test_paths = list_file_paths(os.path.join(test_dir, 'input'))
        self.images = test_paths
        self.label = test_paths

    def __getitem__(self, index):

        single_img = self.images[index]
        single_label = self.label[index]
        img_hist = self.histogram_loader(single_img)
        label_hist = self.histogram_loader(single_label)
        img_hist = torch.Tensor(img_hist)
        label_hist = torch.Tensor(label_hist)
        
        single_img_np = self.images[index]
        
        return img_hist, label_hist, single_img_np

    def __len__(self):
        return len(self.images)
    
class evaset(Dataset):
    def __init__(self, test_dir, result_dir):
        self.images = list_file_paths(os.path.join(test_dir, 'input'))
        self.results = list_file_paths(result_dir)

    def __getitem__(self, index):
        single_img_np = self.images[index]
        single_label_np = self.get_gt_image_path(single_img_np)
        single_result_np = self.results[index]
        
        return single_img_np, single_label_np, single_result_np

    def __len__(self):
        return len(self.images)
    
    def get_gt_image_path(self, input_image_path):
        new_path = re.sub(r'input/([^/]+)/Haze-[123]/', r'gt/\1/', input_image_path)
        new_path = new_path.replace('_synt', '')

        return new_path

def get_training_set():
	train_data  = trainset()
	trainloader = DataLoader(train_data, batch_size=16,shuffle=True)
	
	return trainloader

def get_test_set(test_dir):
	test_data  = testset(test_dir)
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	
	return testloader

def get_eva_set(test_dir, result_dir):
	eva_data  = evaset(test_dir, result_dir)
	evaloader = DataLoader(eva_data, batch_size=1, shuffle=False)
	
	return evaloader