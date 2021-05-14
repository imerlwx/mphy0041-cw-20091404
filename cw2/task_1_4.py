import os
import h5py
import torch
import random
import numpy as np
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
file_path = './dataset70-200.h5'
RESULT_PATH = './result'

## network class
class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):  # in-channel = 1, out-channel = 1, intial features = 32
        super(UNet, self).__init__()

        # down sampling
        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat)
        self.down_shortcut1 = UNet.shortcut(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=[1,0])
        self.encoder2 = UNet._block(n_feat, n_feat*2)
        self.down_shortcut2 = UNet.shortcut(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottom layer
        self.bottleneck = UNet._block(n_feat*2, n_feat*4)
        self.bottleneck_shortcut = UNet.shortcut(n_feat*2, n_feat*4)
        
        # up sampling
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2)
        self.up_shortcut2 = UNet.shortcut((n_feat*2)*2, n_feat*2)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=2, stride=2, padding=[1,0])
        self.decoder1 = UNet._block(n_feat*2, n_feat)
        self.up_shortcut1 = UNet.shortcut(n_feat*2, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)
        self.dropout = torch.nn.Dropout2d(p=0.2)  # one ensemble method
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        # down sampling and shortcut
        enc1 = self.relu(self.encoder1(x) + self.down_shortcut1(x))
        enc2 = self.relu(self.encoder2(self.pool1(enc1)) + self.down_shortcut2(self.pool1(enc1)))
        enc2 = self.dropout(enc2)  

        # bottom layer
        bottleneck = self.relu(self.bottleneck(self.pool2(enc2)) + self.bottleneck_shortcut(self.pool2(enc2)))
        
        # up sampling and skip layer and shortcut
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.relu(self.decoder2(dec2) + self.up_shortcut2(dec2))
        dec2 = self.dropout(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.relu(self.decoder1(dec1) + self.up_shortcut1(dec1))
        return torch.sigmoid(self.conv(dec1))

    # Residual blocks that contain convolutions, non-linear activations, residual shortcuts and one normalization method
    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat)
            )
        
    @staticmethod
    def shortcut(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat)
            )


## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    numerator = torch.sum(y_true*y_pred, dim=(1,2,3)) * 2
    denominator = torch.sum(y_true, dim=(1,2,3)) + torch.sum(y_pred, dim=(1,2,3)) + eps
    return torch.mean(1. - (numerator / denominator))

# a data augmentation method
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
    ])

## data loader
class h5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transforms=None, is_train=True):
        self.h5_file = h5py.File(file_path, 'r')
        self.num_cases = len(set([k.split('_')[1] for k in self.h5_file.keys()]))  # number of cases
        self.num_frames = torch.zeros(self.num_cases, 1)
        for idx in range(self.num_cases):  # number of frames of each case
            self.num_frames[idx] = len([k for k in self.h5_file.keys() if k.split('_')[0]=='frame' if k.split('_')[1]=='%04d' % idx])  

        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return (160 if self.is_train else 40)

    def __getitem__(self, idx):

        if self.is_train:
            idy = random.randint(0, self.num_frames[idx]-1)  # frames in each case have equal chance to be sampled
            idz = torch.randint(0,3,())  # random sample one label from the available three labels
            image = torch.unsqueeze(torch.tensor(self.h5_file["/frame_%04d_%03d" % (idx, idy)][()].astype('float32')), dim=0)
            label1 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, idz)][()].astype('int64')), dim=0)  # one kind of label        

            label20 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 0)][()].astype('int64')), dim=0)
            label21 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 1)][()].astype('int64')), dim=0)
            label22 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 2)][()].astype('int64')), dim=0)
            label2 = (label20 + label21 + label22) / 3

            label2 = (label2>=0.5) * 1  # another kind of label

            if self.transforms is not None:
                image, label1, label2 = self.transforms(image, label1, label2)

            return image, label1, label2
        
        else:
            idx = idx + 160
            idy = random.randint(0, self.num_frames[idx]-1)  # frames in each case have equal chance to be sampled
            idz = torch.randint(0,3,())  # random sample one label from the available three labels
            image = torch.unsqueeze(torch.tensor(self.h5_file["/frame_%04d_%03d" % (idx, idy)][()].astype('float32')), dim=0)

            label0 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 0)][()].astype('int64')), dim=0)
            label1 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 1)][()].astype('int64')), dim=0)
            label2 = torch.unsqueeze(torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 2)][()].astype('int64')), dim=0)
            label = (label0 + label1 + label2) / 3
            label = (label>=0.5) * 1  # another kind of label

            return image, label

def iou(target, pred, batchsize):
    preds = pred.detach().numpy()
    preds = np.array(preds)
    targets = np.array(target)
    iou_score = np.zeros((batchsize,1))
    n = batchsize

    for i in range(batchsize):
      if np.sum(targets[i,0])==0:
        n-=1
      else:
        intersection = np.logical_and(preds[i,0], targets[i,0])
        union = np.logical_or(preds[i,0], targets[i,0])
        iou_score[i] = np.sum(intersection) / np.sum(union)    

    iou_score = np.sum(iou_score)/n

    return iou_score


## training
model = UNet(1,1)  # input 1-channel 2d volume and output 1-channel segmentation (a probability map)
if use_cuda:
    model.cuda()

# training data loader
train_set = h5Dataset(file_path, transforms=None)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=1)

# holdout test data loader
test_set = h5Dataset(file_path, transforms=None, is_train=False)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=True,  # change to False for predefined test data
    num_workers=1)

# optimisation loop
freq_print = 100  # in steps
freq_test = 100  # in steps
total_steps = int(2e4)
step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

while step < total_steps:
    for ii, (images, label1s, label2s) in enumerate(train_loader):
        step += 1
        if use_cuda:
            images, label1s, label2s = images.cuda(), label1s.cuda(), label2s.cuda()

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_dice(preds, label1s)
        #loss = loss_dice(preds, label2s)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if (step % freq_print) == 0:    # print every freq_print mini-batches
            print('Step %d loss: %.5f' % (step,loss.item()))

        # --- testing during training (no validation labels available)
        if (step % freq_test) == 0:  
            images_test, labels_test = iter(test_loader).next()  # test one mini-batch
            if use_cuda:
                images_test = images_test.cuda()
            preds_test = model(images_test)
            iou_score = iou(labels_test, preds_test, 4)
            print('Step %d iou: %.5f' % (iou_score))

print('Training done.')

## save trained model
torch.save(model, 'task1_label1_saved_model_pt')  # label1 model
# torch.save(model, 'task1_label2_saved_model_pt') # label2 model
print('Model saved.')

