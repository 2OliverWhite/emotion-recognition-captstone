print('start')
import torch
print('torch')
from torchvision import datasets, models
print('torchvision')

import torch.nn as nn
print('torch.nn')

import torch.optim as optim
print('torch.optim')

import os 
import numpy as np
import argparse

from ResAttention import *

from dataLoader import get_single_set
from train_evaluate import evaluate as evaluate_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 16
num_classes = 3

print(torch.cuda.is_available())
print(torch.cuda.current_device())
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# state_dict = torch.load('./weights/threeRes50DecayFreeze_epoch_49.pth')
# model.load_state_dict(state_dict)


model = model.to(device)

print('get dataloaders')
root = "./images/cv2/Three/Train"
dataLoader = get_single_set(root=root, input_size=224, batch_size=batch_size)


criterion = nn.CrossEntropyLoss()


res = evaluate_model(model=model, dataloader=dataLoader, criterion=criterion, is_labelled=True, generate_labels=True, k=3 )

for item in res[:-1]:
    print(item)


# plot loss and accuracy
"""
print()
print("Plots of loss and accuracy during training")
print("=" * 20)

x = np.arange(0,50,1)
plt.plot(x, train_losses, label='Training loss')
plt.plot(x, val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(x, train_acc, label='Training accuracy')
plt.plot(x, val_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
"""
