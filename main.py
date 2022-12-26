
from fileIO import *

import os
import time
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential

from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Conv2DTranspose, UpSampling2D, MaxPooling2D 
from keras import optimizers

from keras.layers import Input, concatenate
import keras.backend as K

from sklearn import metrics

SEED = 96

np.random.seed(SEED)

def jaccard_index(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    jaccard = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return jaccard

# def dice_loss(targets, inputs, smooth=1):
    
#     #flatten label and prediction tensors
#     # inputs = K.reshape(inputs, (targets.shape[1], targets.shape[2]))
#     inputs = K.flatten(inputs)
#     inputs = K.reshape(inputs, (targets.shape[1], targets.shape[2]))
#     # print(f'inputs shape at diceLoss = {inputs.shape}')
#     targets = K.flatten(targets)
#     targets = K.reshape(targets, (inputs.shape[0], inputs.shape[1]))
#     # print(f'targets shape at diceLoss = {targets.shape}')
#     # inputs = tf.expand_dims(inputs, -1, name=None)
#     # targets = tf.expand_dims(targets, -1, name=None)
    
#     intersection = K.sum(K.dot(targets, inputs))
#     dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)

#     return 1 - dice

def make_unet(in_shape):

    inputs = Input(shape=in_shape)
    # create Preprocesing layers
    
    # create Model Architecture layers
    x = Conv2D(64, (3, 3), activation = 'relu', padding='same')(inputs)
    copy1 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(x)
    x = MaxPooling2D((2,2))(copy1)

    x = Conv2D(128, (3, 3), activation = 'relu', padding='same')(x)
    copy2 = Conv2D(128, (3, 3), activation = 'relu', padding='same')(x)
    x = MaxPooling2D((2,2))(copy2)

    x = Conv2D(256, (3, 3), activation = 'relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation = 'relu', padding='same') (x)

    z = Conv2DTranspose(128, 3, 2, padding='same')(x)
    z = concatenate([z, copy2])
    z = Conv2D(128, (3,3), activation='relu', padding='same')(z)
    z = Conv2D(128, (3,3), activation='relu', padding='same')(z)

    z = Conv2DTranspose(64, 3, 2, padding='same')(z)
    z = concatenate([z, copy1])
    z = Conv2D(64, (3,3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3,3), activation='relu', padding='same')(z)

    outputs = Conv2D(1, (1,1), activation='sigmoid', padding='same')(z)

    ### Can't print here :(
    # print(f'outputs shape {outputs.shape}')
    # print(f'outputs min {np.min(outputs)}')
    # print(f'outputs mean {np.mean(outputs)}')
    # print(f'outputs max {np.max(outputs)}')

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2, name='iou'), jaccard_index, tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    # model.compile(loss=dice_loss, optimizer=optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2, name='iou'), jaccard_index, tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    # model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2, name='iou'), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


#Calculate pixel accuracy for a segmentation result, compare to groundtruth.
def pix_acc(img, gt):
    img, gt = np.array(img).flatten(), np.array(gt).flatten()
    acc = np.sum(img == gt)

    return acc/len(img)

# def auc(img, gt):
#     img, gt = np.array(img).flatten(), np.array(gt).flatten()
#     auc_score = metrics.roc_auc_score(gt, img)

#     return auc_score

#Calculate intersect over union, IOU = mean(intersect / union)
def iou_coef(y_true, y_pred, smooth=1):
    # intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    # union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    # iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    img, gt = np.array(y_pred).flatten(), np.array(y_true).flatten()
    intersection = np.sum(gt * img)
    union = np.sum(gt + img) - intersection
    iou = np.mean((intersection + smooth)/(union + smooth))

    return iou

#Calcluate Dice coefficient, dice_coef = mean(2 * intersect / (intersect + union))
def dice_coef(img, gt, smooth = 1):
    
    img, gt = np.array(img).flatten(), np.array(gt).flatten()
    numerator = 2 * np.sum(gt * img) + smooth
    denominator = np.sum(gt + img) + smooth

    return np.mean(numerator / denominator)

def model_and_write(x, pred, y_test, imgnum, auc_score, savedir, savename1, savename2):
    # pixel_list, auc_list, iou_list, dice_list = [], [], [], []
    pixel_list, iou_list, dice_list = [], [], []
    totalimg = len(y_test)

    fig, axs = plt.subplots(4, imgnum,figsize=(8,5))

    for i in range(totalimg):
        
        
        mask = pred[i] > 0.5
        # print(f'pred[i].shape={pred[i].shape}')
        # print(f'pred[i] min ={np.min(pred[i])}')
        # print(f'pred[i] mean ={np.mean(pred[i])}')
        # print(f'pred[i] max ={np.max(pred[i])}')
        seg0 = np.zeros(pred[i].shape)
        seg0[mask] = 1
        # print(f'seg0.shape = {seg0.shape}')
        # print(f'seg0 min ={np.min(seg0)}')
        # print(f'seg0 mean ={np.mean(seg0)}')
        # print(f'seg0 max ={np.max(seg0)}')

        if i < imgnum:
            axs[0,i].imshow(x[i], cmap='gray')
            axs[0,i].set_title('input')
            axs[1,i].imshow(y_test[i], cmap='gray')
            axs[1,i].set_title('ground truth')
            axs[2,i].imshow(seg0, cmap='gray')
            axs[2, i].set_title('predicted')

            arr_1 = np.zeros((y_test[i].shape[0],y_test[i].shape[1],3))
            arr_1[...,0] = y_test[i][...,0].copy()
            arr_1[...,1] = seg0[...,0].copy()
            axs[3,i].imshow(arr_1)
            axs[3, i].set_title('R=gt, G=pred')
        
        #Calculate and record pixel accuracy
        pixel_list.append(pix_acc(seg0, y_test[i]))
        # auc_list.append(auc(seg0, y_test[i]))
        #Calculate and record IOU
        iou_list.append(iou_coef(seg0, y_test[i]))
        #Calculate and record dice coefficient
        dice_list.append(dice_coef(seg0, y_test[i]))
    
    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout()
    plt.savefig(savedir + '/' + savename1)

    avg_acc = sum(pixel_list)/totalimg
    # avg_auc = sum(auc_list)/totalimg
    avg_iou = sum(iou_list)/totalimg
    avg_dice = sum(dice_list)/totalimg

    print(savename1 + "pixelacc: " + str(avg_acc) + " iou: " + str(avg_iou) + " dicecoef: " + str(avg_dice))

    X = ['pixelacc', 'auc', 'iou', 'dice']
    scores = [avg_acc, auc_score, avg_iou, avg_dice]

    fig, axs = plt.subplots(1, figsize=(5,5))
    axs.bar(X, scores)
    axs.set_ylim([0, 1.2])
    for (metric, score) in zip(X, scores):
        axs.annotate('%0.3f' % score, (metric, 1.1))
    axs.set_title('Evaluation on test dataset')
    axs.set_ylabel('metric value')
    axs.set_xlabel('metric')
    plt.tight_layout()
    plt.savefig(savedir+ '/' + savename2)



    return


def main():

    size = 'size256'
    EPOCHS=50
    save_suffix = '_unet3keras_epochs'+str(EPOCHS)+'_'+size+'_DICE'
    data_dir = './datasets/'+size+'/'
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_processed_data(data_dir)
    
    imgnum = 5
    in_shape = x_train[0].shape
    model = make_unet(in_shape)
    # model.fit(x_train, y_train, epochs=10, batch_size=1, shuffle=True, validation_split=0.1)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=1, shuffle=True, validation_data = (x_valid,y_valid))
    
    fig, axs = plt.subplots(1, 4, figsize=(10,4))

    
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'val'], loc='upper left')

    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title('model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'val'], loc='upper left')

    axs[2].plot(history.history['iou'])
    axs[2].plot(history.history['val_iou'])
    axs[2].set_title('model iou')
    axs[2].set_ylabel('keras iou')
    axs[2].set_xlabel('epoch')
    axs[2].legend(['train', 'val'], loc='upper left')

    axs[3].plot(history.history['jaccard_index'])
    axs[3].plot(history.history['val_jaccard_index'])
    axs[3].set_title('model iou')
    axs[3].set_ylabel('implemented jaccard')
    axs[3].set_xlabel('epoch')
    axs[3].legend(['train', 'val'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_eval_'+save_suffix+'.jpg')

    loss, accuracy, meanIOU, jaccard, auc, precision, recall = model.evaluate(x_test, y_test, verbose=0)

    print(f'loss = {loss}')
    print(f'accuracy = {accuracy}')
    print(f'meanIOU = {meanIOU}')
    print(f'jaccard = {jaccard}')
    print(f'auc = {auc}')
    print(f'precision = {precision}')
    print(f'recall = {recall}')

    
    pred = model.predict(x_test)

    SAVE_DIR = 'output_images_deep_seg'
    save_name1 = 'predictions_'+save_suffix
    save_name2 = 'test_eval_'+save_suffix
    model_and_write(x_test, pred, y_test, imgnum, auc, SAVE_DIR, save_name1, save_name2)
    
    return

if __name__ == '__main__':
	main()