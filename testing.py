"""
Author: Zachary Wimpee

Description: Script for evaluating trained model performance
             on holdout test set data.
             
Note: Currently only produces the calculated performance metrics
      on holdout data. Next step is to also return visualizations for 
      model interpretibility.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from architecture import ResidualNet, ConvNet
from sklearn.metrics import classification_report
from tqdm import tqdm

# Define tuple for class clabels. 
classes = ('COVID', 'Normal', 'Pneumonia')

def get_loader(batch_size, pin_memory, num_workers):
    # Get path to test set folder.
    test_path = Path('data/test').resolve()
    
    # Read mean and std from json file.
    with open('mean_std.json') as f:
        mean_std = json.load(f)

    mean = mean_std['mean']
    std = mean_std['std']
    
    # Define the transforms.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    
    return DataLoader(ImageFolder(test_path,transform),
                      batch_size=batch_size,
                      pin_memory=pin_memory,
                      num_workers=num_workers)

def get_model(arch, state_dict, device):
    if arch == "res":
        model = ResidualNet()
    else:
        model = ConvNet()
    
    path = '/'.join(['models',state_dict])
    if device.type == "cuda":
        model.load_state_dict(torch.load(path))
        model.to(device)
    else:
        model.load_state_dict(torch.load(path, map_location=device))
        
    return model

def main(args):
    # Assign device specified by user.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Get pretrained model.
    model = get_model(args.arch,args.state_dict,device)
    
    # Get test dataloader.
    test_dl = get_loader(args.batch_size,args.memory_pinning,args.workers)
    
    # Initialize tensors to store test outputs.
    preds = torch.tensor([],device=device)
    targets = torch.tensor([],device=device)
    
    # Set model to evaluation mode predict on test data.
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = torch.cat((targets,labels),dim=0)
            outputs = model(inputs)
            preds = torch.cat((preds, outputs.argmax(dim=1)),dim=0)
            
    # Print out the classification report.
    if use_cuda:
        print(classification_report(targets.cpu(),preds.cpu()))
    else:
        print(classification_report(targets,preds))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument("-c","--use-cuda", action="store_true", help="try to use GPU")
    parser.add_argument("-a", "--arch", type=str, default="res", choices=["res", "conv"], help="architecture of pretrained model")
    parser.add_argument("-s", "--state-dict", type=str, default="trained_model.pt", help="pretrained model filename")
    parser.add_argument("-b","--batch-size", type=int, default=128, help="test dataloader batch size")
    parser.add_argument("-p", "--memory-pinning", action="store_true", help="copy tensors to CUDA pinned memory")
    parser.add_argument("-w","--workers",type=int, default=4, help="number of subprocesses for data loading")
    args = parser.parse_args()
    main(args)
    
        