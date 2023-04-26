
import matplotlib.pyplot as plt
import numpy as np
import re
import glob

def scale_to_unit(arr):
    return np.interp(arr, (arr.min(), arr.max()), [0, +1])

def figure_from_results(path):
    with open(path, 'r') as f:
        f.readline() # Train Loss
        train_losses = f.readline()
        f.readline() # Train Acc
        train_acc = f.readline()
        f.readline() # Val Loss
        val_losses = f.readline()
        f.readline() # Val Acc 
        val_acc = f.readline()
        
        # Train Loss
        train_losses = train_losses.split(',')[:-1] # Split Into Parts
        train_losses[0] = train_losses[0][1:] # Remove Starting Bracket
        train_losses = [float(loss) for loss in train_losses]

        # Val Loss
        val_losses = val_losses.split(',')[:-1] # Split Into Parts
        val_losses[0] = val_losses[0][1:] # Remove Starting Bracket
        val_losses = [float(loss) for loss in val_losses]

        # Train Acc
        train_acc = re.sub("tensor\((.*?),.*?\)", r"\1", train_acc);
        train_acc = train_acc.split(',')[:-1] # Split Into Parts
        train_acc[0] = train_acc[0][1:] # Remove Starting Bracket
        train_acc = [float(loss) for loss in train_acc]

        # Val Acc
        val_acc = re.sub("tensor\((.*?),.*?\)", r"\1", val_acc);
        val_acc = val_acc.split(',')[:-1] # Split Into Parts
        val_acc[0] = val_acc[0][1:] # Remove Starting Bracket
        val_acc = [float(loss) for loss in val_acc]

        val_acc = np.array(val_acc)
        train_acc = np.array(train_acc)
        val_losses = np.array(val_losses)
        train_losses = np.array(train_losses)

        val_losses = scale_to_unit(val_losses)
        train_losses = scale_to_unit(train_losses)

        x = np.arange(0,50,1)
        plt.plot(x, train_losses, label='Training loss')
        plt.plot(x, val_losses, label='Validation loss')
        plt.plot(x, train_acc, label='Training Acc')
        plt.plot(x, val_acc, label='Validation Acc')
        plt.legend(frameon=False)
        plt.title("Loss - Acc Graph")
        plt.xlabel("Epoch")
        plt.ylabel("Y")
        plt.savefig(f"{path.split('/')[-1]}.png")
        plt.clf()


for path in glob.glob("./losses_multi/*"):
    figure_from_results(path)
import sys
sys.exit()


x = np.arange(0,50,1)
plt.plot(x, train_loss, label='Training loss')
plt.plot(x, val_loss, label='Validation loss')
plt.legend(frameon=False)
plt.title("Resnet")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(x, train_acc, label='Training accuracy')
plt.plot(x, val_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.title(" Resnet")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

