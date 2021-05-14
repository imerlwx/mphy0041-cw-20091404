import os
import h5py
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
file_path = './dataset70-200.h5'


## network class
class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):  # in-channel = 1, out-channel = 1, intial features = 32
        super(UNet, self).__init__()

        # down sampling
        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat)
        self.down_shortcut1 = UNet.down_shortcut(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=[1,0])
        self.encoder2 = UNet._block(n_feat, n_feat*2)
        self.down_shortcut2 = UNet.down_shortcut(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottom layer
        self.bottleneck = UNet._block(n_feat*2, n_feat*4)
        self.bottleneck_shortcut = UNet.down_shortcut(n_feat*2, n_feat*4)
        
        # up sampling
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2)
        self.up_shortcut2 = UNet.up_shortcut((n_feat*2)*2, n_feat*2)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=2, stride=2, padding=[1,0])
        self.decoder1 = UNet._block(n_feat*2, n_feat)
        self.up_shortcut1 = UNet.up_shortcut(n_feat*2, n_feat)

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
    def down_shortcut(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat)
            )
        
    @staticmethod
    def up_shortcut(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat)
            )

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

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel("Average of 2 measures")
    plt.ylabel("Difference between 2 measures")
    plt.title('Bland-Altman Plot')
    plt.show()
    plt.savefig('Bland-Altman-Plot.png')

# holdout test data loader
test_set = h5Dataset(file_path, transforms=None, is_train=False)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=40,
    shuffle=True,  # change to False for predefined test data
    num_workers=1)

net = torch.load('task2_saved_model_pt')  # classification
net2 = torch.load('task1_label2_saved_model_pt') # segmentation

for ii, (images, labels) in enumerate(test_loader):
  images, labels = iter(test_loader).next()  # test one mini-batch
  outputs = net(images)
  outputs_without = net2(images)

diff_without = torch.zeros(40,1) 
diff_with = torch.zeros(40,1)
images_with = torch.zeros(40,1,58,52)
labels_with = torch.zeros(40,1,58,52)
mean = torch.zeros(40,1)
diff = torch.zeros(40,1)

_, predicted = torch.max(outputs, 1)  # make the output numbers to 0 and 1

# only put the images that have nonzero label into inputs
for i in range(40):
  if predicted[i]==1:
    images_with[i,0] = images[i,0]
    labels_with[i,0] = labels[i,0]

outputs_with = net2(images_with)

# difference with or without the classification
for i in range(40):
    diff_without[i] = (outputs_without[i] - labels[i]).sum()
    diff_with[i] = (outputs_with[i] - labels_with[i]).sum()
    mean[i] = (diff_with[i] + diff_without[i]) / 2
    diff[i] = diff_with[i] - diff_without[i]

diff_without = diff_without.detach().numpy()
diff_with = diff_with.detach().numpy()
mean = mean.detach().numpy()
diff = diff.detach().numpy()

bland_altman_plot(diff_without, diff_with)