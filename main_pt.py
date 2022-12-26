import torch
import torch.nn as nn
import torch.optim as optim

from model import *
from fileIO_pt import *

import os
import time
import matplotlib.pyplot as plt

SEED = 96

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main():

    size = 'size512'
    data_dir = './datasets/'+size+'/'

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_processed_data(data_dir)

    print(f'Number of train imgs = {len(x_train)}, Number of validation imgs = {len(x_valid)}, Number of test imgs = {len(x_test)}')

    print(f'train img = {x_train[0].shape}, train label = {y_train[0].shape}')
    print(f'valid img = {x_valid[0].shape}, valid label = {y_valid[0].shape}')
    print(f'test img = {x_test[0].shape}, test label = {y_test[0].shape}')

    # for i, (tr_i, tr_gt, v_i, v_gt, te_i, te_gt) in enumerate(zip(x_train, y_train, x_valid, y_valid, x_test, y_test)):
    #     print(f'train img {i} = {tr_i.shape}, train label {i} = {tr_gt.shape}')
    #     print(f'valid img {i} = {v_i.shape}, valid label {i} = {v_gt.shape}')
    #     print(f'test img {i} = {te_i.shape}, test label {i} = {te_gt.shape}')
        
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = UNet()
    
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    ### Training ###
    EPOCHS = 50
    save_suffix = '_unet2015pytorch_epochs'+str(EPOCHS)+'_'+size+'_BCE'
    SAVE_DIR = 'models'
    SAVE_PATH = os.path.join(SAVE_DIR, "mini-unet.pt")

    time_curr = time.time()
    best_val_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(EPOCHS):
        train_loss = train(model, device, x_train, y_train, optimizer, criterion)
        train_losses.append(train_loss)
        valid_loss = evaluate(model, device, x_valid, y_valid, criterion, epoch)
        valid_losses.append(valid_loss)

        # if valid_loss < best_val_loss:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         }, SAVE_PATH)
        #     best_val_loss = valid_loss
            
        print('| Epoch: {0:d} | Train Loss: {1:.4f} | Valid Loss : {2:.4f} | Time: {3:d}'.format(epoch+1, 
            train_loss, valid_loss, int(time.time() - time_curr)))
        time_curr = time.time()
    
    # Visualie
    X = np.linspace(0, EPOCHS, EPOCHS)
    fig = plt.figure()
    plt.title("Training Deep Learning model")
    plt.plot(X,train_losses, label='training data')
    plt.plot(X,valid_losses, label='validation data')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f'training_loss_'+save_suffix+'.jpg')

    ### Testing ### 
    model.eval()
    test_loss = evaluate(model, device, x_test, y_test, criterion, EPOCHS+1)
    fname = f'segPT_'+save_suffix+'.jpg'
    imgnum = 5
    Acc, AuC, IoU, Dice = predict(model, device, x_test, y_test, imgnum, 0.5, fname)
    print('| Test Loss: {0:.4f} | Avg Acc: {0:.4f} | Avg AuC: {0:.4f} | Avg IoU: {0:.4f} | Avg Dice: {0:.4f} |'.format(test_loss, Acc, AuC, IoU, Dice))

    X = ['pixelacc', 'auc', 'iou', 'dice']
    scores = [Acc, AuC, IoU, Dice]
    save_name2 = 'test_eval_'+save_suffix

    fig, axs = plt.subplots(1, figsize=(5,5))
    axs.bar(X, scores)
    axs.set_ylim([0, 1.2])
    for (metric, score) in zip(X, scores):
        axs.annotate('%0.3f' % score, (metric, 1.1))
    axs.set_title('Evaluation on test dataset')
    axs.set_ylabel('metric value')
    axs.set_xlabel('metric')
    plt.tight_layout()
    plt.savefig(save_name2)

    return

if __name__ == '__main__':
	main()