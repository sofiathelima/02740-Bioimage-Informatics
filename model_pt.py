import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
# import torchmetrics
# from torchmetrics import JaccardIndex
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import os

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)

        self.conv11 = nn.Conv2d(2,1,1)

        self.pool = nn.MaxPool2d(2)

        self.convt6 = nn.ConvTranspose2d(1024,512,1, stride=2)
        self.convt7 = nn.ConvTranspose2d(512,256,1, stride=2)
        self.convt8 = nn.ConvTranspose2d(256,128,1, stride=2)
        self.convt9 = nn.ConvTranspose2d(128,64,1, stride=2)

        self.convt1 = nn.ConvTranspose2d(1024,512,3)
        self.convt2 = nn.ConvTranspose2d(512,256,3)
        self.convt3 = nn.ConvTranspose2d(256,128,3)
        self.convt4 = nn.ConvTranspose2d(128,64,3)
        self.convt5 = nn.ConvTranspose2d(64,2,1)
    
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        copy1 = x
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        copy2 = x
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        copy3 = x
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        copy4 = x
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        return x, copy1, copy2, copy3, copy4
    
    def crop(self, copy, z):
        (_, H, W) = z.shape
        copy = CenterCrop([H, W])(copy)
        return copy
    
    def decoder(self, x, copy1, copy2, copy3, copy4):
        
        # print(f'Decode step 1')
        # print(f'x shape = {x.shape}')
        z = F.relu(self.convt6(x))
        # print(f'z shape = {z.shape}')
        # print(f'copy4 shape = {copy4.shape}')
        white = self.crop(copy4, z)
        # print(f'white shape = {white.shape}')
        z = torch.cat([z, white], dim=0)
        z = F.relu(self.convt1(z))
        z = F.relu(self.conv8(z))
        
        z = F.relu(self.convt7(z))
        white = self.crop(copy3, z)
        z = torch.cat([z, white], dim=0)
        z = F.relu(self.convt2(z))
        z = F.relu(self.conv6(z))

        z = F.relu(self.convt8(z))
        white = self.crop(copy2, z)
        z = torch.cat([z, white], dim=0)
        z = F.relu(self.convt3(z))
        z = F.relu(self.conv4(z))

        z = F.relu(self.convt9(z))
        white = self.crop(copy1, z)
        z = torch.cat([z, white], dim=0)
        z = F.relu(self.convt4(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.convt5(z))

        z = self.conv11(z)


        # z = torch.tanh(z)
        z = torch.sigmoid(z)
        # z = F.softmax(z)



        return z
    
    def forward(self,x):
        
        z, copy1, copy2, copy3, copy4 = self.encoder(x)
        x = self.decoder(z, copy1, copy2, copy3, copy4)

        # pd = (0,1)
        # x = F.pad(x, pd, "constant", 0)

        # print(f'x size before conv11= {x.shape}')

        # x = self.conv11(x)

        # print(f'x size after conv11 = {x.shape}')

        # print(f'x max = {np.max(x.cpu().detach().numpy())}')
        # print(f'x mean = {np.mean(x.cpu().detach().numpy())}')
        # print(f'x min = {np.min(x.cpu().detach().numpy())}')

        # x = torch.add(x[0], x[1])
        # x = torch.sum(x, dim=0)
        
        return x

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train(model, device, imgs, lbls, optimizer, criterion):

    epoch_loss = 0
    
    model.train()
    
    for (x, y) in zip(imgs, lbls):
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        fx = model(x)

        # print(f'fx shape = {fx.shape}, y shape = {y.shape}')
        # print(f'fx[1,:,:] = {fx[1,:,:].shape}')
        # print(f'fx[1] = {fx[1].shape}')
        # print(f'fx[1:2] = {fx[1:2].shape}')


        y = CenterCrop([fx.shape[1], fx.shape[2]])(y)
        
        loss = criterion(fx, y)

        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    
    return epoch_loss / len(imgs)

def evaluate(model, device, imgs, lbls, criterion, epoch):
    
    epoch_loss = 0

    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in zip(imgs, lbls):

            x = x.to(device)
            y = y.to(device)

            fx = model(x)
            
            y = CenterCrop([fx.shape[1], fx.shape[2]])(y)
            

            loss = criterion(fx, y)

            # acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            # epoch_acc += acc.item()
    

    # # Visualize histograms at each epoch to ensure that feature map values are between 0 and 1

    # fig, axs = plt.subplots(2,2, figsize=(20,7))

    # x = imgs[0]
    # x = x.to(device)
    # fx = model(x)
    # fx = fx.cpu().detach().numpy()

    # axs[0,0].imshow(fx[0])
    # axs[0,0].set_title('fx')
    # axs[0,1].hist(fx[0].flatten())
    # axs[0,0].set_title('fx histogram')

    # y = lbls[0]
    # y = CenterCrop([fx.shape[1], fx.shape[2]])(y)
    # y = y.cpu().detach().numpy()

    # axs[1,0].imshow(y[0])
    # axs[1,0].set_title('y')
    # axs[1,1].hist(y[0].flatten())
    # axs[1,1].set_title('y histogram')

    # plt.tight_layout()
    # plt.savefig(os.path.join('histograms', f'epoch{epoch}.jpg'))
        
    return epoch_loss / len(imgs)

#Calculate pixel accuracy for a segmentation result, compare to groundtruth.
def pix_acc(img, gt):
    img, gt = np.array(img).flatten(), np.array(gt).flatten()
    acc = np.sum(img == gt)

    return acc/len(img)

def auc(img, gt):
    img, gt = np.array(img).flatten(), np.array(gt).flatten()
    auc_score = metrics.roc_auc_score(gt, img)

    return auc_score

def jaccard_index(img, gt, smooth=1):
    img, gt = np.array(img).flatten(), np.array(gt).flatten()

    numerator = np.sum(gt * img) + smooth
    denominator = (np.sum(gt + img)-numerator) + smooth

    return np.mean(numerator / denominator)

#Calcluate Dice coefficient, dice_coef = mean(2 * intersect / (intersect + union))
def dice_coef(img, gt, smooth = 1):
    
    img, gt = np.array(img).flatten(), np.array(gt).flatten()
    numerator = 2 * np.sum(gt * img) + smooth
    denominator = np.sum(gt + img) + smooth

    return np.mean(numerator / denominator)

def predict(model, device, imgs, lbls, imgnum, THRESHOLD, fname):
    
    fig, axs = plt.subplots(4,imgnum, figsize=(8,5))

    Acc = 0
    AuC = 0
    IoU = 0
    Dice = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(zip(imgs, lbls)):

            x = x.to(device)
            y = y.to(device)

            fx = model(x)
            mask = fx > THRESHOLD
            seg = torch.zeros(fx.shape)
            seg[mask] = 1
            pred = seg.cpu().detach().numpy()
            
            img = CenterCrop([fx.shape[1], fx.shape[2]])(x)
            img = img.cpu().detach().numpy()
            y = CenterCrop([fx.shape[1], fx.shape[2]])(y)
            y = y.type(torch.uint8)
            target = y.cpu().detach().numpy()

            if i < imgnum:

                axs[0, i].imshow(img[0], cmap='gray')
                axs[0, i].set_title('input')
                axs[1, i].imshow(target[0], cmap='gray')
                axs[1, i].set_title('ground truth')
                axs[2, i].imshow(pred[0], cmap='gray')
                axs[2, i].set_title('predicted')

                
                # target = np.moveaxis(target, -1, 0)

                arr_1 = np.zeros((target.shape[1],target.shape[2],3))
                arr_1[...,0] = target[0].copy()
                arr_1[...,1] = pred[0].copy()
                axs[3, i].imshow(arr_1)
                axs[3, i].set_title('R=gt, G=pred')

            Acc += pix_acc(pred, target)
            AuC += auc(pred, target)
            IoU += jaccard_index(pred, target)
            Dice += dice_coef(pred, target)

    [axi.set_axis_off() for axi in axs.ravel()]
    plt.tight_layout()
    plt.savefig(fname)

    return Acc / len(imgs), AuC / len(imgs), IoU / len(imgs), Dice / len(imgs)