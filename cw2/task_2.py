import os
import random

import torch
import h5py
import torchvision.models as models
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
filename = './dataset70-200.h5'
RESULT_PATH = './result'

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
        image = torch.unsqueeze(torch.tensor(self.h5_file["/frame_%04d_%03d" % (idx, idy)][()].astype('float32')), dim=0)        

        vote = 0
        
        for idz in range(3):

            labelmap = torch.tensor(self.h5_file["/label_%04d_%03d_%02d" % (idx, idy, idz)][()].astype('int64'))
            if labelmap.sum() == 0:
                vote += 1
            else:
                vote += 0

        if vote >= 2:
            label = 0
        else:
            label = 1 

        return image, label

## adapt and train a densenet121 network class
model = models.densenet121(num_classes=2)
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

if use_cuda:
    model.cuda()

train_set = h5Dataset(filename)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8, 
    shuffle=True,
    num_workers=8)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

freq_print = 10
for epoch in range(200):
    for step, (frames, labels) in enumerate(train_loader, 0):
        if use_cuda:
            frames, labels = frames.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if step % freq_print == (freq_print-1):    # print every 20 mini-batches
            print('[Epoch %d, iter %05d] loss: %.3f' % (epoch, step, loss.item()))
            moving_loss = 0.0

print('Training done.')

## save trained model
torch.save(model, os.path.join(RESULT_PATH,'saved_model_pt'))  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
print('Model saved.')
