import os

import torch
import numpy as np


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
        self.encoder1 = torch.nn.ReLU(UNet._block(ch_in, n_feat) + UNet.shortcut(ch_in, n_feat))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = torch.nn.ReLU(UNet._block(n_feat, n_feat*2) + UNet.shortcut(n_feat, n_feat*2))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = torch.nn.ReLU(UNet._block(n_feat*2, n_feat*4) + UNet.shortcut(n_feat*2, n_feat*4))
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = torch.nn.ReLU(UNet._block(n_feat*4, n_feat*8) + UNet.shortcut(n_feat*4, n_feat*8))
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottom layer
        self.bottleneck = torch.nn.ReLU(UNet._block(n_feat*8, n_feat*16) + UNet.shortcut(n_feat*8, n_feat*16))
        
        # up sampling
        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder4 = torch.nn.ReLU(UNet._block((n_feat*8)*2, n_feat*8) + UNet.shortcut((n_feat*8)*2, n_feat*8))
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder3 = torch.nn.ReLU(UNet._block((n_feat*4)*2, n_feat*4) + UNet.shortcut((n_feat*4)*2, n_feat*4))
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = torch.nn.ReLU(UNet._block((n_feat*2)*2, n_feat*2) + UNet.shortcut((n_feat*2)*2, n_feat*2))
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=2, stride=2)
        self.decoder1 = torch.nn.ReLU(UNet._block(n_feat*2, n_feat) + UNet.shortcut(n_feat*2, n_feat))

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):

        # down sampling
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # bottom layer
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # up sampling and skip layer
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    # Residual blocks that contain convolutions, non-linear activations, residual shortcuts and one normalization method
    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat))

    @staticmethod
    def shortcut(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat))


## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(1. - (numerator / denominator))

## data loader
class h5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')
        self.num_cases = len(set([k.split('_')[1] for k in self.h5_file.keys()]))  # number of cases
        self.num_frames = torch.zeros(num_cases, 1)
        for idx in range(self.num_cases):  # number of frames of each case
            self.num_frames[idx] = len([k for k in self.h5_file.keys() if k.split('_')[0]=='frame' if k.split('_')[1]=='%04d' % idx])  

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        
        idy = random.randint(0, self.num_frames[idx]-1)  # frames in each case have equal chance to be sampled
        idz = torch.randint(0,2,())  # random sample one label from the available three labels

        image = torch.unsqueeze(torch.tensor(self.h5_file["/frame_%04d_%03d" % (idx, idy)][()].astype('float32')), dim=0)
        label1 = torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, idz)][()].astype('int64'))  # one kind of label        

        label20 = torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 0)][()].astype('int64'))
        label21 = torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 1)][()].astype('int64'))
        label22 = torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, 2)][()].astype('int64'))
        label2 = (label20 + label21 + label22) / 3

        label2 = (label2>=0.5) * 1  # another kind of label

        return image, label1, label2

## training
model = UNet(1,1)  # input 1-channel 2d volume and output 1-channel segmentation (a probability map)
if use_cuda:
    model.cuda()

# training data loader
train_set = h5Dataset(file_path)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=4)

# optimisation loop
freq_print = 100  # in steps
#freq_test = 2000  # in steps
total_steps = int(2e5)
step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

while step < total_steps:
    for ii, (images, label1s, label2s) in enumerate(train_loader):
        step += 1
        if use_cuda:
            images, label1s, label2s = images.cuda(), label1s.cuda(), label2s.cuda()

        optimizer.zero_grad()
        preds = model(images)
        loss1 = loss_dice(preds, label1s)
        loss1.backward()
        loss2 = loss_dice(preds, label2s)
        loss2.backward()
        optimizer.step()

        # Compute and print loss
        if (step % freq_print) == 0:    # print every freq_print mini-batches
            print('Step %d loss1: %.5f loss2: %.5f' % (step,loss1.item(),loss2.item()))

print('Training done.')

## save trained model
torch.save(model, os.path.join(RESULT_PATH,'saved_model_pt')) 
print('Model saved.')

