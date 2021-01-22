###############################################################################
# CXResnet model training
#
# Author: Zachary Wimpee
#
# This file contains the procedures for training a model using one of the 
# networks defined in the architecture.py file. It can be executed using the
# default command line arguments, but many values can be specified by the user
# allowing for increased flexibility and versatility.
# 
#
###############################################################################
import argparse
from pathlib import Path
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from preprocessing import rmtree

from architecture import ResidualNet, ConvNet


classes = ('COVID','Normal','Pneumonia')


def imshow_raw(img, mean, std):
    """Show unnormalized image tensor."""
    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

def n_random_samples(dataset, n_samples = 100):
    # Get n_samples number of dataset indices.
    n = torch.randperm(len(dataset))[:n_samples].tolist()
    
    images = torch.stack([dataset[i][0] for i in n])
    labels = torch.tensor([dataset[i][1] for i in n])
    
    return images, labels

def prediction_probs(model, images):
    # Get model outputs for input images.
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        
        # Get index of predicted class.
        preds = outputs.argmax(dim=1).tolist()
    
        # Get the probability of the predictions.
        probs = [F.softmax(outputs[i],dim=0)[preds[i]].item() for i in range(len(preds))]
        
    return preds, probs

def show_predictions(model, images, labels, mean, std):
    # Get predicted labels and the probabilities.
    preds, probs = prediction_probs(model, images)
    
    # Get the size of the mini-batch.
    size = len(images)
    
    # Initialize figure for showing images and predictions.
    fig = plt.figure(figsize=(24,24*size))
    
    # Plot each image and color by prediction outcome.
    for i in range(size):
        ax = fig.add_subplot(1,size,i+1,xticks=[],yticks=[])
        imshow_raw(images[i], mean, std)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[i]],
            probs[i] * 100.0,
            classes[labels[i]]),
                    color=("green" if preds[i]==labels[i].item() else "red"))
    return fig


# Printing out the results after each epoch.
def epoch_end(epoch, train_scores, val_scores):
    print(
        'Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
        .format(epoch+1,train_scores['train_loss'],train_scores['train_acc'],val_scores['val_loss'],val_scores['val_acc']))
    
        
# Training function for each epoch.
def epoch_phase(model,criterion,optimizer,
                loader,device, phase,
                epoch, running_loss, writer):
    # Initialize values to track.
    i = epoch * len(loader) # current number of minibatches for phase
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
        # Increment minibatch count.
        i += 1
        
        # Get input and label batch, append labels to targets. 
        inputs = inputs.to(device,non_blocking=True)
        labels = labels.to(device,non_blocking=True)
        targets = torch.cat((targets,labels),dim=0)

        # Zero the optimizer gradients.
        optimizer.zero_grad()
        
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

        # Increment running loss
        running_loss += loss.item()
        
        # Deavgerage and increment the epoch training loss.
        epoch_loss += loss.item() * inputs.size(0)
        
        # Log the phase's running loss every 10 minibatches.
        if i % 10 == 9:
            writer.add_scalar('Loss/Batches/{}'.format(phase), running_loss / 10, i)
            running_loss = 0.0
            
        
        
    # Calculate the training loss and accuracy score.
    epoch_loss = epoch_loss / len(loader.dataset)
    
    counts = torch.unique(targets, return_counts=True)[1]
    epoch_acc = torch.tensor([torch.logical_and(torch.eq(targets, i),torch.eq(preds, i)).sum() / counts[i] for i in range(3)]).sum() / 3

    # Save epoch scores in dict.
    epoch_scores = {'{}_loss'.format(phase): epoch_loss,
                    '{}_acc'.format(phase): epoch_acc,
                    '{}_rl'.format(phase): running_loss}
    
    return epoch_scores

def main(args):
    # Assign device as GPU if available, else CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            transforms.RandomHorizontalFlip(p=0.33),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
    
    valset = torchvision.datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
    
    
    train_dl = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                           shuffle=True,pin_memory=args.memory_pinning,num_workers=args.workers)
    
    val_dl = torch.utils.data.DataLoader(valset,batch_size=2*args.batch_size,
                                         shuffle=False,pin_memory=args.memory_pinning, num_workers=args.workers)
    
    # Initialize model, loss function, and optimizer.
    if args.arch == "res":
        model = ResidualNet().to(device)
    else:
        model = ConvNet().to(device)
        
    criterion = nn.CrossEntropyLoss()
    
    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
    # Clear tensorboard logs if specified.
    if args.tb_clear:
        rmtree(Path('runs'))
    
    # Create run folder for tensorboard.
    run = '/'.join(['runs', args.tb_folder])
    if Path(run).is_dir(): # clear run if it exists
        rmtree(Path(run))    
    writer = SummaryWriter(run)
    
    # Train for specified number of epochs.
    best_acc = 0.0
    train_rl = 0.0
    val_rl = 0.0
    #best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(args.epochs):
        train_scores = epoch_phase(model, criterion, optimizer, loader=train_dl, device=device, phase='train', epoch=epoch, running_loss=train_rl, writer=writer)
        val_scores = epoch_phase(model, criterion, optimizer, loader=val_dl, device=device, phase='val', epoch=epoch, running_loss=val_rl, writer=writer)
        
        # Update the running losses.
        train_rl = train_scores['train_rl']
        val_rl = val_scores['val_rl']
        
        epoch_end(epoch, train_scores, val_scores)
        
        # Write loss and accuracy scores to TensorBoard.
        writer.add_scalar('Loss/Epoch/train', train_scores['train_loss'], epoch)
        writer.add_scalar('Loss/Epoch/val', val_scores['val_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_scores['train_acc'], epoch)
        writer.add_scalar('Accuracy/val', val_scores['val_acc'], epoch)
        
        # Write some validation images and predictions to TensorBoard.
        model.eval()
        with torch.no_grad():
            # Randomly select 4 validation samples.
            inputs, labels = n_random_samples(valset, 4)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Plot the sample predictions and write to TensorBoard.
            writer.add_figure(
                'sample prediction outcomes',
                show_predictions(
                    model,inputs,
                    labels,mean,std),global_step=epoch)
            writer.flush()
        
        """if val_scores['val_acc'] > best_acc:
            best_acc = val_scores['val_acc']
            best_weights = copy.deepcopy(model.state_dict())"""
    
            
    # Save the trained model to specified path.
    #model.load_state_dict(best_weights)
    PATH = '/'.join(['models',args.save_path])
    torch.save(model.state_dict(), PATH)
    writer.close()
    

if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument("-d","--data", type=str, default="data", help="name of data directory")
    parser.add_argument("-b","--batch-size", type=int, default=128, help="dataloader batch size")
    parser.add_argument("-p", "--memory-pinning", action="store_true", help="copy tensors to CUDA pinned memory")
    parser.add_argument("-w","--workers",type=int, default=4, help="number of subprocesses for data loading")
    parser.add_argument("-a","--arch", type=str, default="res", choices = ["res","conv"], help="select model architecture")
    parser.add_argument("-o","--optim", type=str, default="adam", choices=["adam","sgd"],help="select optimizer algorithm")
    parser.add_argument("-l","--lr", type=float, default=0.001,help="optimizer learning rate")
    parser.add_argument("-m","--momentum", type=float, help="momentum parameter value, required when not using adam optimizer")
    parser.add_argument("-e","--epochs",type=int, default=10,help="number of epochs to train")
    parser.add_argument("-s","--save-path",type=str,default="trained_model.pt", help="path to save trained model")
    parser.add_argument("-c","--tb-clear", action="store_true", help="clears all tensorboard logs")
    parser.add_argument("-t","--tb-folder", type=str, default="cxresnet", help="tensorboard child directory for model being trained")
    args = parser.parse_args()
    main(args)
        
    
