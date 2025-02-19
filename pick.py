import os
import shutil
import random
from datasets import list_file_paths

source_dir = './data/DTB70'
target_dir = './data/DTB70-Haze'
gt_main_dir = './data/DTB70-GT'
gt_sub_dirs = os.listdir(gt_main_dir)

num_to_copy = int(len(list_file_paths(gt_main_dir)) * 0.8)
copied_count = 0

random.shuffle(gt_sub_dirs)

for sub_dir in gt_sub_dirs:
    if copied_count < num_to_copy:
        action_type = 'train'
        copied_count += len(list_file_paths(os.path.join(gt_main_dir, sub_dir)))
    else:
        action_type = 'test'

    shutil.copytree(os.path.join(gt_main_dir, sub_dir), os.path.join(target_dir, action_type, 'gt', sub_dir))
    shutil.copytree(os.path.join(source_dir, 'DTB70-1', sub_dir), os.path.join(target_dir, action_type, 'input', 'Haze-1', sub_dir))
    shutil.copytree(os.path.join(source_dir, 'DTB70-2', sub_dir), os.path.join(target_dir, action_type, 'input', 'Haze-2', sub_dir))
    shutil.copytree(os.path.join(source_dir, 'DTB70-3', sub_dir), os.path.join(target_dir, action_type, 'input', 'Haze-3', sub_dir))

    print(f"sub directory {sub_dir} copied")

print(f"finish copying {copied_count} sub directories from source to target ( include gt, h-1, h-2 and h-3 )")

