from torchvision import transforms, datasets
import torch



def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight        

def get_dataloaders(train_root, valid_root, input_size, batch_size, shuffle=True):
    '''
    Create the dataloaders for train, validation and test set. Randomly rotate images for data augumentation
    Normalization based on std and mean.
    '''
    data_transform = transforms.Compose(
                                        [transforms.RandomRotation(25),
                                         transforms.Resize(input_size),
                                         transforms.CenterCrop(input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    workers = 4
    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform)
    val_dataset = datasets.ImageFolder(root=valid_root, transform=data_transform)


    train_weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))                                                                
    train_weights = torch.DoubleTensor(train_weights)                                       
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))  

    val_weights = make_weights_for_balanced_classes(val_dataset.imgs, len(val_dataset.classes))                                                                
    val_weights = torch.DoubleTensor(val_weights)                                       
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_weights))  

    dataLoader = {'train': torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                            sampler=train_sampler,
                                             num_workers=workers), 
                  'valid': torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                            sampler=val_sampler,
                                             num_workers=workers)}
    return dataLoader


def get_single_set(root, input_size, batch_size):
    data_transform = transforms.Compose(
                                        [transforms.RandomRotation(25),
                                         transforms.Resize(input_size),
                                         transforms.CenterCrop(input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    workers = 4
    dataset = datasets.ImageFolder(root=root, transform=data_transform)


    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))  



    dataLoader =  torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=sampler, num_workers=workers)
    return dataLoader