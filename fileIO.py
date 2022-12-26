import os
import re
import numpy as np
from skimage.io import imread
import tensorflow as tf
import keras


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
            train_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(train_path, f))
            train_gts.append(gt)
    
    img_shape1 = train_imgs[0].shape[0]
    img_shape2 = train_imgs[0].shape[1]
    img_shape3 = train_imgs[0].shape[2]

    gt_shape1 = train_imgs[0].shape[0]
    gt_shape2 = train_imgs[0].shape[1]
    gt_shape3 = train_imgs[0].shape[2]

    x_train = np.zeros((len(train_imgs), img_shape1, img_shape2, img_shape3))
    y_train = np.zeros((len(train_gts), gt_shape1, gt_shape2, gt_shape3))
    for i in range(x_train.shape[0]):
        x_train[i] = train_imgs[i]
        y_train[i] = train_gts[i]
    x_train = x_train / np.max(x_train)
    y_train = y_train / np.max(y_train)

    # Get Validation data
    ##########################################
    valid_path = os.path.join(data_dir, 'validation')
    valid_files = os.listdir(valid_path)
    valid_imgs = []
    valid_gts = []
    for f in valid_files:
        if re.search('image', f):
            img = imread(os.path.join(valid_path, f))
            valid_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(valid_path, f))
            valid_gts.append(gt)
    
    img_shape1 = valid_imgs[0].shape[0]
    img_shape2 = valid_imgs[0].shape[1]
    img_shape3 = valid_imgs[0].shape[2]

    gt_shape1 = valid_gts[0].shape[0]
    gt_shape2 = valid_gts[0].shape[1]
    gt_shape3 = valid_gts[0].shape[2]

    x_valid = np.zeros((len(valid_imgs), img_shape1, img_shape2, img_shape3))
    y_valid = np.zeros((len(valid_gts), gt_shape1, gt_shape2, gt_shape3))
    for i in range(x_valid.shape[0]):
        x_valid[i] = valid_imgs[i]
        y_valid[i] = valid_gts[i]
    x_valid = x_valid / np.max(x_valid)
    y_valid = y_valid / np.max(y_valid)
    
    # Get Testing data
    ##########################################
    test_path = os.path.join(data_dir, 'testing')
    test_files = os.listdir(test_path)
    test_imgs = []
    test_gts = []
    for f in test_files:
        if re.search('image', f):
            img = imread(os.path.join(test_path, f))
            test_imgs.append(img)
        elif re.search('label', f):
            gt = imread(os.path.join(test_path, f))
            test_gts.append(gt)
    
    img_shape1 = test_imgs[0].shape[0]
    img_shape2 = test_imgs[0].shape[1]
    img_shape3 = test_imgs[0].shape[2]

    gt_shape1 = test_gts[0].shape[0]
    gt_shape2 = test_gts[0].shape[1]
    gt_shape3 = test_gts[0].shape[2]

    x_test = np.zeros((len(test_imgs), img_shape1, img_shape2, img_shape3))
    y_test = np.zeros((len(test_gts), gt_shape1, gt_shape2, gt_shape3))
    for i in range(x_test.shape[0]):
        x_test[i] = test_imgs[i]
        y_test[i] = test_gts[i]
    x_test = x_test / np.max(x_test)
    y_test = y_test / np.max(y_test)


    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_raw_datasets():

    # dir_path = './datasets'

    # Get data from STARE (Hoover et al. 2000)
    ##########################################
    # Get Images
    stare_data_path = 'datasets/STARE/stare-images'
    stare_data_fnames = os.listdir(stare_data_path)
    stare_imgs = []

    for i, f in enumerate(stare_data_fnames):
        # im reads in channel last
        img = imread(os.path.join(stare_data_path, f), plugin='matplotlib')
        img = img[...,1:2]
        img = tf.image.resize(img, (128, 128))
        stare_imgs.append(img)
    

    # Get Labels
    stare_labelVK_path = 'datasets/STARE/labels-vk'
    stare_label_fnames = os.listdir(stare_labelVK_path)
    stare_gts = []

    for i, f in enumerate(stare_label_fnames):
        img = imread(os.path.join(stare_labelVK_path, f))
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        img = tf.image.resize(img, (128, 128))
        stare_gts.append(img)
    

    # Get data from DRIVE (Staal et al. 2004)
    # ##########################################
    # Get Images
    drive_train_data_path = 'datasets/DRIVE/training/images'
    drive_train_fnames = os.listdir(drive_train_data_path)
    drive_imgs = []

    for i, f in enumerate(drive_train_fnames):
        img = imread(os.path.join(drive_train_data_path, f), plugin='matplotlib')
        img = img[...,1:2]
        img = tf.image.resize(img, (128, 128))
        drive_imgs.append(img)

    # Get Labels
    drive_train_label_path = 'datasets/DRIVE/training/1st_manual'
    drive_train_label_fnames = os.listdir(drive_train_label_path)
    drive_gts = []

    for i, f in enumerate(drive_train_label_fnames):
        img = imread(os.path.join(drive_train_label_path, f))
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        img = tf.image.resize(img, (128, 128))
        drive_gts.append(img)
    

    # Get data from CHASE
    ##########################################
    # Images and Labels
    chase_data_path = 'datasets/CHASEDB1'
    chase_files = os.listdir(chase_data_path)
    
    chase_imgs = []
    chase_fnames = []

    for i, f in enumerate(chase_files):
        if re.search('.jpg',f):
            img = imread(os.path.join(chase_data_path, f))
            img = img[...,1:2]
            img = tf.image.resize(img, (128, 128))
            chase_imgs.append(img)
            chase_fnames.append(f)

    chase_gts1 = []
    chase_gts2 = []

    for f in chase_fnames:
        sample_num = f.split('_')[-1].split('.')[0]
        for f2 in chase_files:
            if re.search(sample_num, f2) and re.search('1st',f2):
                gt1 = imread(os.path.join(chase_data_path, f2))
                gt1 = np.resize(gt1, (gt1.shape[0], gt1.shape[1], 1))
                gt1 = tf.image.resize(gt1, (128, 128))
                chase_gts1.append(gt1)
            elif re.search(sample_num, f2) and re.search('2nd', f2):
                gt2 = imread(os.path.join(chase_data_path, f2))
                gt2 = np.resize(gt2, (gt2.shape[0], gt2.shape[1], 1))
                gt2 = tf.image.resize(gt2, (128, 128))
                chase_gts2.append(gt2)
    


    imgs = np.concatenate((stare_imgs, drive_imgs, chase_imgs), axis=0)
    gts = np.concatenate((stare_gts, drive_gts, chase_gts1), axis=0)

    rand_indxs = np.random.choice(len(imgs), size=len(imgs), replace=False)

    train_size = 0.8
    train_imgs = np.zeros((int(len(imgs)*train_size), imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    train_gts = np.zeros((int(len(imgs)*train_size), imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))

    for i, ri in enumerate(rand_indxs[0:int(len(imgs)*train_size)]):
        train_imgs[i] = imgs[ri]
        train_gts[i] = gts[ri]
    train_imgs = train_imgs / np.max(train_imgs)
    train_gts = train_gts / np.max(train_gts)
    

    test_indxs = rand_indxs[int(len(imgs)*train_size):]
    test_imgs = np.zeros((int(len(test_indxs))+1, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    test_gts = np.zeros((int(len(test_indxs))+1, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))

    for i, ri in enumerate(test_indxs):
        test_imgs[i] = imgs[ri]
        test_gts[i] = gts[ri]
    test_imgs = test_imgs / np.max(test_imgs)
    test_gts = test_gts / np.max(test_gts)
    
    
    return train_imgs, train_gts, test_imgs, test_gts