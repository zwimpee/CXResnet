"""
Author: Zachary Wimpee

Description: Script for training a model and saving 
             the state_dict for the best weights.
             
Note: Should figure out how to implement command line
      arguments to generalize the training script to 
      any desired architecture and specify training 
      parameters. 
"""
import argparse
from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import architecture
from sklearn.metrics import balanced_accuracy_score

# Printing out the results after each epoch.
def epoch_end(epoch, train_scores, val_scores):
    print(
        'Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
        .format(epoch+1,
                train_scores['train_loss'],train_scores['train_acc'],
                val_scores['val_loss'],val_scores['val_acc']))
    
        
# Training function for each epoch.
def epoch_phase(model, criterion, optimizer, loader, device, phase):
    # Initialize values to track.
    epoch_loss = 0.0
    preds = torch.tensor([],device=device)
    targets = torch.tensor([],device=device)
    
    # Set model to phase mode.
    if phase == 'train':
        model.train()
    else:
        model.eval()
        
    # Loop over dataloader batches.
    for inputs, labels in loader:
        # Get input and label batch, append labels to targets. 
        inputs = inputs.to(device,non_blocking=True)
        labels = labels.to(device,non_blocking=True)
        targets = torch.cat((targets,labels),dim=0)

        # Zero the optimizer gradients.
        #optimizer.zero_grad()
        model.zero_grad()
        
        with torch.set_grad_enabled(phase == 'train'):
            # Get outputs and calculate the predicted class label indices.
            outputs = model(inputs)
            preds = torch.cat((preds, outputs.argmax(dim=1)),dim=0)

            # Calulate the batch loss, and do backpropagation.
            loss = criterion(outputs,labels)
            
            if phase == 'train':
                # Do backpropagation and update the model weights.
                loss.backward()
                optimizer.step()
            
        

        # Increment the epoch training loss (scaled by batch size).
        epoch_loss += loss.item() * inputs.size(0)
        
    # Calculate the training loss and accuracy score.
    epoch_loss = epoch_loss / len(loader.dataset)
    epoch_acc = balanced_accuracy_score(targets.cpu(),preds.cpu())

    # Save epoch scores in dict.
    epoch_scores = {'{}_loss'.format(phase): epoch_loss,
                    '{}_acc'.format(phase): epoch_acc}
    
    return epoch_scores



def main(args):
    # Assign device as GPU if available, else CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Print GPU information if using cuda.
    if device.type == 'cuda':
        print('-'*20)
        print(torch.cuda.get_device_name(0))
        print('Total Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
        print('Memory allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Memory cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('-'*10)
    
    # Loading training and validation data.
    traindir = (Path(args.data)/'train').resolve()
    valdir = (Path(args.data)/'val').resolve()
    
    # Read mean and std from json file.
    with open('mean_std.json') as f:
        mean_std = json.load(f)

    mean = mean_std['mean']
    std = mean_std['std']
    
    trainset = torchvision.datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
    
    train_dl = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                           shuffle=True,pin_memory=True)
    
    val_dl = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])),
        batch_size=2*args.batch_size,shuffle=False,pin_memory=True)
    
    # Initialize model, loss function, and optimizer.
    model = architecture.ResidualNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train for specified number of epochs.
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        train_scores = epoch_phase(model, criterion, optimizer, loader=train_dl, device=device, phase='train')
        val_scores = epoch_phase(model, criterion, optimizer, loader=val_dl, device=device, phase='val')
        
        epoch_end(epoch, train_scores, val_scores)
        
        if val_scores['val_acc'] > best_acc:
            best_acc = val_scores['val_acc']
            best_weights = copy.deepcopy(model.state_dict())
            
    # Save the trained model to specified path.
    model.load_state_dict(best_weights)
    PATH = args.save_path
    torch.save(model, PATH)
    

if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument("--data", type=str, default="data", help="name of data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="dataloader batch size")
    parser.add_argument("--lr", type=float, default=0.001,help="optimizer learning rate")
    parser.add_argument("--epochs",type=int, default=10,help="number of epochs to train")
    parser.add_argument("--save-path",type=str,default="trained_model.pt", help="path to save trained model")
    args = parser.parse_args()
    main(args)
        
    
