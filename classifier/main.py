print('start')
import torch
print('torch')
from torchvision import datasets
from torchvision import models
print('torchvision')

import torch.nn as nn
print('torch.nn')

import torch.optim as optim
print('torch.optim')

import os 
import numpy as np
import argparse

from ResAttention import *

from dataLoader import get_dataloaders
from train_evaluate import train_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", help="number of predicted classes",
                    type=int, default="3")
parser.add_argument("--batch_size", help="batch size of cnn",
                    type=int, default="16")
parser.add_argument("--num_epochs", help="training epochs",
                    type=int, default="50")
parser.add_argument("--train_root", help="training data directory",
                    type=str, default="./images/Three/Train/")
parser.add_argument("--valid_root", help="validation data directory",
                    type=str, default="./images/Three/Valid/")
parser.add_argument("--name", help="weights output name",
                    type=str, default="three")
parser.add_argument("--freeze", help="weights output name",
                    type=str, default="False")
parser.add_argument("--type", help="weights output name",
                    type=int, default=50)
parser.add_argument("--decay", help="alter optim weight decay",
                    type=str, default="False")

args = parser.parse_args()

print(torch.cuda.is_available())
print(torch.cuda.current_device())

batch_size = 16
shuffle_datasets = True
num_epochs = 50
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)
save_all_epochs = True

weight_decay = 0

if args.decay == "True":
    weight_decay = 1e-3

print('load model')
# model = ResidualAttentionModel_92(num_classes = args.num_classes)

model = None

if args.type == 50:
    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
elif args.type == 101:
    model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
else:
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)


if args.freeze == "True":
    for param in model.parameters():
        param.requires_grad = False


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, args.num_classes)
#model = model.load_state_dict(torch.load('weights/fullset.pth'))

model = model.to(device)


print('get dataloaders')
dataLoader = get_dataloaders(train_root=args.train_root, valid_root=args.valid_root, input_size=224, batch_size=args.batch_size, shuffle=shuffle_datasets)

print('dataloader')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)

# training
print("Training progress")
print("=" * 20)
trained_model, train_losses, train_acc, val_losses, val_acc = train_model(model=model, dataloaders=dataLoader, criterion=criterion, optimizer=optimizer, name=args.name, save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=args.num_epochs)
#vids = []
#for filename in glob.glob('../train/Positive/*.mp4'):
#    vids.append(filename)
#res = evaluate(model=model, dataloaders=vids)
# save the model
print(train_losses)
print(train_acc)
print(val_losses)
print(val_acc)
torch.save(trained_model.state_dict(), f"weights/best_{args.name}_{args.num_epochs}.pth")

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
