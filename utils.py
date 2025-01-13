import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def hist_match(input_img_path, label_img_path, output_R_PDF, output_G_PDF, output_B_PDF):   
    input_img = cv2.imread(input_img_path)
    label_img = cv2.imread(label_img_path)
    result = np.copy(input_img)
    input_R_hist, _ = np.histogram(input_img[:, :, 2], bins=256, range=(0, 256)) 
    input_G_hist, _ = np.histogram(input_img[:, :, 1], bins=256, range=(0, 256))
    input_B_hist, _ = np.histogram(input_img[:, :, 0], bins=256, range=(0, 256))
    input_B = input_img[:,:,0]
    input_G = input_img[:,:,1]
    input_R = input_img[:,:,2]
    input_B_shape = input_B.shape    
    input_G_shape = input_G.shape 
    input_R_shape = input_R.shape 
    input_B = input_B.ravel()
    input_G = input_G.ravel()
    input_R = input_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        _, i_bin_idx_B, i_counts_B = np.unique(input_B, return_inverse=True,return_counts=True)
        _, i_bin_idx_G, i_counts_G = np.unique(input_G, return_inverse=True,return_counts=True)
        _, i_bin_idx_R, i_counts_R = np.unique(input_R, return_inverse=True,return_counts=True)

        i_quantiles_B = np.cumsum(i_counts_B).astype(np.float64)
        i_quantiles_B /= i_quantiles_B[-1]
        i_quantiles_G = np.cumsum(i_counts_G).astype(np.float64)
        i_quantiles_G /= i_quantiles_G[-1]
        i_quantiles_R = np.cumsum(i_counts_R).astype(np.float64)
        i_quantiles_R /= i_quantiles_R[-1]
        o_quantiles_B = np.cumsum((output_B_PDF.squeeze(0)*sum(input_B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum((output_G_PDF.squeeze(0)*sum(input_G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum((output_R_PDF.squeeze(0)*sum(input_R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        interp_t_values_B = np.interp(i_quantiles_B, o_quantiles_B, values) #, b_values
        interp_t_values_G = np.interp(i_quantiles_G, o_quantiles_G, values)
        interp_t_values_R = np.interp(i_quantiles_R, o_quantiles_R, values)
        result[:,:,0] = interp_t_values_B[i_bin_idx_B].reshape(input_B_shape)
        result[:,:,1] = interp_t_values_G[i_bin_idx_G].reshape(input_G_shape)
        result[:,:,2] = interp_t_values_R[i_bin_idx_R].reshape(input_R_shape)

        return result, label_img
    except ValueError:
        pass

def npTOtensor(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    
    return image

def align_to_four(img):

    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]

    return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

