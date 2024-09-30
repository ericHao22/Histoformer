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
        for file in sorted(files):
            file_list.append(os.path.join(root, file))
    return file_list

train_dir = './data/DTB70_Haze/train/'
train_paths = list_file_paths(train_dir)

gt_dir = './data/DTB70_Haze/gt/'
gt_paths = list_file_paths(gt_dir)

test_dir = './data/DTB70_Haze/test/'
test_paths = list_file_paths(test_dir)

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
        self.label = gt_paths

    def __getitem__(self, index):

        single_img = self.images[index]
        single_label = self.label[index]
        img_hist = self.histogram_loader(single_img)
        label_hist = self.histogram_loader(single_label)
        
        img_hist = torch.tensor(img_hist,dtype=torch.float)#.permute(1,0)#.unsqueeze(1) [3,256]
        label_hist = torch.tensor(label_hist,dtype=torch.float)#.permute(1,0)#.unsqueeze(1) [3,256]
        
        single_img_np = self.images[index]        
        single_label_np = self.label[index]
        
        return img_hist, label_hist, single_img_np, single_label_np#ori_img, hs_img 

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader

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


def get_training_set():
	train_data  = trainset()
	trainloader = DataLoader(train_data, batch_size=16,shuffle=True)
	
	return trainloader

def get_test_set():
	test_data  = testset()
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	
	return testloader