from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from datasets import get_eva_set
import cv2
import argparse

def avg(score):
    return sum(score)/len(score)

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_path", type=str, default="./data/Drone-Haze/test/", help='Path to test data')
parser.add_argument("--result_path", type=str, default="./results/Drone-Haze/", help='Path to saved result')
opt = parser.parse_args()

evaloader = get_eva_set(opt.test_data_path, opt.result_path)

before_psnr_list = []
before_ssim_list = []
after_psnr_list = []
after_ssim_list = []

for i,(test_path, gt_path, result_path) in enumerate(evaloader):
    gt_img = cv2.imread(gt_path[0])
    test_img = cv2.imread(test_path[0])
    result_img = cv2.imread(result_path[0])
    before_psnr = psnr(gt_img, test_img)
    before_ssim = ssim(gt_img, test_img, multichannel=True)
    after_psnr = psnr(gt_img, result_img)
    after_ssim = ssim(gt_img, result_img, multichannel=True)
    before_psnr_list.append(before_psnr)
    before_ssim_list.append(before_ssim)
    after_psnr_list.append(after_psnr)
    after_ssim_list.append(after_ssim)
    
    print(f'file name:{gt_path[0]}, PSNR_B:{before_psnr}, SSIM_B:{before_ssim}, PSNR_A:{after_psnr}, SSIM_A:{after_ssim}')

print("Before Avgrage PSNR:", avg(before_psnr_list))
print("Before Avgrage SSIM:", avg(before_ssim_list))
print("After Average PSNR:", avg(after_psnr_list))
print("After Average SSIM", avg(after_ssim_list))


