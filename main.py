# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations,
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 16
max_num_epoch = 100
hps = {'lr':0.0067}
num_layers_pool = [1,2,4]
kernel_size_pool = [3,5]
num_kernel_pool = [2,4,8]


# ---- options ----
DEVICE_ID = 'cuda' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

# --- imports ---
import torch
import os
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils

torch.multiprocessing.set_start_method('spawn', force=True)


def run_experiments(lr=-1.0,ks = -1,nk=-1,nl = -1):
    random.seed()
    device = torch.device(DEVICE_ID)
    print('device: ' + str(device))
    print('Hyperparameter optimization begins')
    for i in range(30):

        hps['lr'] = 10 ** random.uniform(-1, -4) if lr ==-1 else lr
        kernel_size = random.choice(kernel_size_pool) if ks == -1 else ks
        num_kernel = random.choice(num_kernel_pool) if nk == -1 else nk
        num_layers = random.choice(num_layers_pool) if nl == -1 else nl
        net = Net(num_layers, kernel_size, num_kernel).to(device=device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
        train_loader, val_loader = get_loaders(batch_size, device)

        if LOAD_CHKPT:
            print('loading the model from the checkpoint')
            model.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))

        print('Starting Training with Parameters : KS %d NL %d NK %d lr %f' %
              (kernel_size, num_layers, num_kernel, hps['lr']))
        max_acc = -1
        prev_acc = -1
        same_cnt = 0
        dec = 0
        best_epoch = -1
        mses = []
        margin12errs = []
        for epoch in range(max_num_epoch):
            running_loss = 0.0  # training loss of the network
            for iteri, data in enumerate(train_loader, 0):
                inputs, targets,path = data  # inputs: low-resolution images, targets: high-resolution images.

                optimizer.zero_grad()  # zero the parameter gradients

                # do forward, backward, SGD step
                preds = net(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                # print loss
                running_loss += loss.item()
                print_n = 100  # feel free to change this constant
                if iteri % print_n == (print_n - 1):  # print every print_n mini-batches
                    print('[%d, %5d] network-loss: %.3f' %
                          (epoch + 1, iteri + 1, running_loss / 100))
                    running_loss = 0.0
                    # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

                if (iteri == 0) and VISUALIZE:
                    hw3utils.visualize_batch(inputs, preds, targets)
            acc = 0
            mses.append(running_loss)
            for iteri, data in enumerate(val_loader, 0):
                inputs_val, targets_val,path = data
                preds_val_raw = net(inputs_val)
                preds_val = preds_val_raw.clone().reshape(1, -1).squeeze()
                targets_flat = targets_val.reshape(1, -1).squeeze()
                a = torch.abs(preds_val - targets_flat)
                d = torch.gt(a, 12 / 127)
                acc += 1-(d.sum().item() / targets_flat.shape[0])
            acc /= (iteri + 1)
            acc *= 127/128
            margin12errs.append(1-acc)
            print('End of epoch %d and accuracy is %.2f' % (epoch + 1, acc))
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            '''
            # If the acc change is less than 0.01 incr count
            if round(abs(acc - prev_acc),2) < 0.01:
                same_cnt += 1
            else:
                same_cnt = 0
            '''
            # Sum the % decrease values
            if round(acc - prev_acc, 2) <= -0.01:
                dec += acc - prev_acc

            # If the same value occured for 15 epochs, early terminate
            if same_cnt == 20:
                print('Early terminating due to repetition at  %d repet_cnt %d', (epoch, same_cnt))
                break
            # If the percentage decrease is more than 20%, early terminate
            elif round(dec, 2) < -0.20:
                print('Early terminating due to error dec at %d', (epoch))
                break
            # Designated max_acc with the epoch number
            if round(acc - max_acc, 2) >= 0.01:
                max_acc = acc
                dec = 0
                best_epoch = epoch
                same_cnt = 0
            else:
                same_cnt += 1
            prev_acc = acc
            # torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint.pt'))
        #hw3utils.visualize_batch(inputs_val,preds_val_raw,targets_val,os.path.join(LOG_DIR,'example.png'))
        # Write the result into a .txt file
        print('Finished Training with Parameters : KS %d NL %d NK %d lr %f : Acc %f Epoch %d' %
              (kernel_size, num_layers, num_kernel, hps['lr'], max_acc, best_epoch))
        f = open('trainings.txt', 'a+')
        strbuf = 'Parameters: KS %d NL %d NK %d lr %f : Acc: %f Epoch : %d\n' \
                 % (kernel_size, num_layers, num_kernel, hps['lr'], max_acc, best_epoch)
        f.write(strbuf)
        f.close()
        hw3utils.visualize_batch(inputs_val,preds_val_raw,targets_val,os.path.join(LOG_DIR,'example.png'))
        # Plots
        plt.close()
        plt.plot(range(1,best_epoch+2),mses[0:best_epoch+1])
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.xlabel('Epochs')
        plt.ylabel('Mean-Squared Error')
        plt.savefig('mse-epochs.png')
        plt.close()
        plt.plot(range(1, best_epoch + 2), margin12errs[0:best_epoch + 1])
        #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.xlabel('Epochs')
        plt.ylabel('12-Margin Error')
        plt.savefig('val-margin.png')

        # Map the predictions back to -255 to 255 range
        imgs = []
        f = open('filenames.txt', 'w+')
        for iteri, data in enumerate(val_loader, 0):
            inp, tg,path = data
            pred = net(inp)
            for img in pred:
                imgs.append(hw3utils.back_to_img_format(img))
            for p in path[0]:
                f.write(p+'\n')

        np.save('predictions',imgs)
        f.close()
        # Break the loop if the parameters are already given
        if ks != -1:
            break
# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset'
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self,num_layers,kernel_size,num_kernels):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 3, 5, padding=2)
        pad = int((kernel_size-1)/2)
        if num_layers == 1:
            self.model = nn.Sequential(
                nn.Conv2d(1,3,kernel_size,padding=pad)
            )
        elif num_layers == 2:
            self.model = nn.Sequential(
                nn.Conv2d(1, num_kernels, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(num_kernels,3,kernel_size,padding=pad)
            )
        elif num_layers == 4:
            self.model = nn.Sequential(
                nn.Conv2d(1, num_kernels, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(num_kernels, 3, kernel_size, padding=pad),
            )
    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = self.model(grayscale_image)
        return x

# ---- training code -----
# For determining hyperparameters
#run_experiments()
# For the further experiments
run_experiments(0.085912,3,16,2)


