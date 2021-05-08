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
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')
        self.num_cases = len(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.num_frames = len([k for k in self.h5_file.keys() if k.split('_')[0]=='frame'])
        self.num_labels = len([k for k in self.h5_file.keys() if k.split('_')[0]=='label'])
    
    def __len__(self):
        return self.num_cases
        
    def __getitem__(self, index):
        idx_frame = random.randint(0, self.num_frames[index]-1)
        frame = torch.unsqueeze(torch.tensor(self.h5_file['/subject%06d_frame%08d' % (index, idx_frame)][()].astype('float32')), dim=0)
        label = torch.squeeze(torch.tensor(self.h5_file['/subject%06d_label%08d' % (index, idx_frame)][()].astype('int64')))
        return (frame, label)

## adapt and train a densenet121 network class
#def DenseNet121()
model = models.DenseNet121()

train_set = H5Dataset(filename)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8, 
    shuffle=True,
    num_workers=8)
'''
dataiter = iter(train_loader)
frames, labels = dataiter.next()
'''

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

freq_print = 10
for epoch in range(2):
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
