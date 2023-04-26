import torch
from tqdm import tqdm
import time
import copy
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
import json

def train_model(model, dataloaders, criterion, optimizer, name, save_dir = None, save_all_epochs=False, num_epochs=30):
    '''
    model: The NN to train
    dataloaders: A dictionary containing at least the keys 
                 'train','val' that maps to Pytorch data loaders for the dataset
    criterion: The Loss function
    optimizer: The algorithm to update weights 
               (Variations on gradient descent)
    num_epochs: How many epochs to train for
    save_dir: Where to save the best model weights that are found, 
              as they are found. Will save to save_dir/weights_best.pth
              Using None will not write anything to disk
    save_all_epochs: Whether to save weights for ALL epochs, not just the best
                     validation error epoch. Will save to save_dir/weights_e{#}.pth
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # record loss and correct
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':

                if epoch % 10 == 9:
                    torch.save(model.state_dict(), f"weights/{name}_epoch_{epoch}.pth")

                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
            if phase == "train":
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)

        

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    with open(f'losses_files_{name}.txt', 'w') as filehandle:
        filehandle.write("Train Losses\n")
        filehandle.write("[")
        for value in train_losses:
            filehandle.write(str(value))
            filehandle.write(",")
        filehandle.write("]")

        filehandle.write("\n")

        filehandle.write("Train Acc\n")
        filehandle.write("[")
        for value in train_acc:
            filehandle.write(str(value))
            filehandle.write(",")
        filehandle.write("]")

        filehandle.write("\n")

        filehandle.write("Val Losses\n")
        filehandle.write("[")
        for value in val_losses:
            filehandle.write(str(value))
            filehandle.write(",")
        filehandle.write("]")

        filehandle.write("\n")

        filehandle.write("Val Acc\n")
        filehandle.write("[")
        for value in val_acc:
            filehandle.write(str(value))
            filehandle.write(",")
        filehandle.write("]")

        filehandle.write("\n")
        



    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, train_acc, val_losses, val_acc

def evaluate(model, dataloader, criterion, is_labelled = False, generate_labels = True, k = 5):
    
    # Evaluation of the model on validation and test set only. (criteria: loss, top1 acc, top5 acc)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0
    running_top1_correct = 0
    running_top5_correct = 0
    predicted_labels = []
    
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        tiled_labels = torch.stack([labels.data for i in range(k)], dim=1) 

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if is_labelled:
                loss = criterion(outputs, labels)

            _, preds = torch.topk(outputs, k=k, dim=1)
            if generate_labels:
                nparr = preds.cpu().detach().numpy()
                predicted_labels.extend([list(nparr[i]) for i in range(len(nparr))])

        if is_labelled:
            running_loss += loss.item() * inputs.size(0)
            running_top1_correct += torch.sum(preds[:, 0] == labels.data)
            running_top5_correct += torch.sum(preds == tiled_labels)
        else:
            pass

    if is_labelled:
        epoch_loss = float(running_loss / len(dataloader.dataset))
        epoch_top1_acc = float(running_top1_correct.double() / len(dataloader.dataset))
        epoch_top5_acc = float(running_top5_correct.double() / len(dataloader.dataset))
    else:
        epoch_loss = None
        epoch_top1_acc = None
        epoch_top5_acc = None

    return epoch_loss, epoch_top1_acc, epoch_top5_acc, predicted_labels

data_transform = transforms.Compose([transforms.RandomRotation(25),
                                     transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def extract(model, dataloader, save_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for vid in dataloader:
            cap = cv2.VideoCapture(vid)
            vname = vid.split('/')[-1] # Split filename from path (./hello/video.mp4 => video.mp4)
            vname = vname.split('.')[0] # Cutoff file extension (video.mp4 => video)
            cnt = 0
            res = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # if cnt % 5 == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                tensor = data_transform(im_pil)
                tensor = tensor.view(1, 3, 224, 224)
                tensor = tensor.to(device)
                output = model(tensor)
                output = output.cpu().numpy()
                res.append(output)
                #cnt += 1
            cap.release()
            print(np.shape(res))
            print('{} feature saved'.format(vname))
            continue
            np.save('{}/{}.npy'.format(save_dir, vname), res)
