import os
import re
import numpy as np
from skimage.io import imread
import torch
import torchvision.transforms as transforms

SEED = 96

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def load_processed_data(data_dir):

    # Get Training data
    ##########################################
    train_path = os.path.join(data_dir, 'training')
    train_files = os.listdir(train_path)

    train_imgs = []
    train_gts = []
    for f in train_files:
        if re.search('image', f):
            img = imread(os.path.join(train_path, f))
            img = np.moveaxis(img, -1, 0)
            img = torch.tensor(img, dtype=torch.float32)
            # img = transforms.functional.equalize(img)
            train_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(train_path, f))
            gt = np.moveaxis(gt, -1, 0)
            # gt = torch.tensor(gt, dtype=torch.uint8)
            gt = torch.tensor(gt, dtype=torch.float32)
            train_gts.append(gt)

    # Get Validation data
    ##########################################
    valid_path = os.path.join(data_dir, 'validation')
    valid_files = os.listdir(valid_path)
    valid_imgs = []
    valid_gts = []
    for f in valid_files:
        if re.search('image', f):
            img = imread(os.path.join(valid_path, f))
            img = np.moveaxis(img, -1, 0)
            # img = torch.tensor(img, dtype=torch.uint8)
            img = torch.tensor(img, dtype=torch.float32)
            valid_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(valid_path, f))
            gt = np.moveaxis(gt, -1, 0)
            # gt = torch.tensor(gt, dtype=torch.uint8)
            gt = torch.tensor(gt, dtype=torch.float32)
            valid_gts.append(gt)
    
    # Get Testing data
    ##########################################
    test_path = os.path.join(data_dir, 'testing')
    test_files = os.listdir(test_path)
    test_imgs = []
    test_gts = []
    for f in test_files:
        if re.search('image', f):
            img = imread(os.path.join(test_path, f))
            img = np.moveaxis(img, -1, 0)
            # img = torch.tensor(img, dtype=torch.uint8)
            img = torch.tensor(img, dtype=torch.float32)
            test_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(test_path, f))
            gt = np.moveaxis(gt, -1, 0)
            # gt = torch.tensor(gt, dtype=torch.uint8)
            gt = torch.tensor(gt, dtype=torch.float32)
            test_gts.append(gt)
    
    
    return train_imgs, train_gts, valid_imgs, valid_gts, test_imgs, test_gts
