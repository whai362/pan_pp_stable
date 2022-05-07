import mmcv
import numpy as np
import random
import torch

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
EPS = 1e-6

rctw_17_root = '../data/RCTW-17/'
rctw_17_train_root = rctw_17_root + 'train/'

img_names = [img_name for img_name in mmcv.utils.scandir(rctw_17_train_root, '.jpg')]

# img_names = np.random.permutation(img_names)


def list_to_str(image_list):
    list_str = ''
    for image in image_list:
        list_str += image + '\n'

    return list_str

# print(img_names)

train_list = img_names[:7346]
val_list = img_names[7346:]

with open(rctw_17_root + 'full_train_list.txt', 'w') as f:
    f.write(list_to_str(img_names))

with open(rctw_17_root + 'train_list.txt', 'w') as f:
    f.write(list_to_str(train_list))

with open(rctw_17_root + 'val_list.txt', 'w') as f:
    f.write(list_to_str(val_list))
