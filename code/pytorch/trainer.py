from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


plt.ion()   # interactive mode


# Load dataset from numpy arrays
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))#.transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=1, fine_tune=False):
    since = time.time()
    criterion = nn.NLLLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_MSE = 1000000.
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_MSE = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(torch.log(outputs), torch.max(labels, 1)[1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_MSE += mse(outputs, labels) #torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_MSE / dataset_sizes[phase]

            print('{} Loss: {:.4f} MSE: {:.4f}'.format(
                phase, epoch_loss, epoch_mse))

            # deep copy the model
            if phase == 'val' and epoch_mse > best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_mse))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

    
dirname = "data_hybrids_fixed_even"

# Add more transforms here... also might want to use ImageNet10 so we don't upsample so much...
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

data = np.load(dirname + "/x_train.npy")
targets = np.load(dirname + "/y_train.npy")
train_dataset = MyDataset(data, targets, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
targets = np.load(dirname + "/y_train_old.npy")
train_dataset = MyDataset(data, targets, transform=transform)
train_prob_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
data = np.load(dirname + "/x_test.npy")
targets = np.load(dirname + "/y_test.npy")
test_dataset = MyDataset(data, targets, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
targets = np.load(dirname + "/y_test_old.npy")
test_dataset = MyDataset(data, targets, transform=transform)
test_prob_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

dataloader_bool = {'train': train_dataloader, 'val': test_dataloader}
dataloader_prob = {'train': train_prob_dataloader, 'val': test_prob_dataloader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Make the model--could change False -> True... might work better...
model_conv = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 8),
    nn.Dropout(0.5),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

for i in range(100):
    # Train the full network
    for param in model_conv.parameters():
        param.requires_grad = True

    model_conv = model_conv.to(device)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1, last_epoch=i-1)

    summary(model_conv, (3, 64, 64))
    
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, dataloader_bool, num_epochs=1, fine_tune=False)


    # Now pause the training of the initial layers
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc.reset_parameters()

    model_conv = model_conv.to(device)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.5)

    summary(model_conv, (3, 64, 64))
    
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, dataloader_prob, num_epochs=25, fine_tune=True)
