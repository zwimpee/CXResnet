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

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from architecture import ResidualNet, ConvNet
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_lightning.metrics.functional import (
    confusion_matrix,
    precision_recall_curve, auc,
    f1, roc, average_precision,
)
from pytorch_lightning.metrics.functional.classification import (
    accuracy,
    multiclass_auroc,
    recall,
    stat_scores_multiple_classes,
)
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set();


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
    probs = torch.tensor([], device=device)
    #preds = torch.tensor([],device=device)
    targets = torch.tensor([],device=device)
    
    # Set model to evaluation mode predict on test data.
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = torch.cat((targets,labels),dim=0)
            outputs = model(inputs)
            probs = torch.cat((probs, F.softmax(outputs, 1)),dim=0)
            
            
    print('Accuracy - {0:.2f} %'.format(accuracy(probs.argmax(dim=1), targets).item()*100.0))
    print('Balanced Accuracy - {0:.2f} %'.format(recall(probs, targets, num_classes=3, class_reduction='macro').item()*100.0))
    print('Balanced Accuracy Weighted - {0:.2f} %'.format(recall(probs, targets, num_classes=3, class_reduction='weighted').item()*100.0))
    print('ROC AUC - {:.4f}'.format(multiclass_auroc(probs, targets).item()))
    print('Macro F-1 - {:.4f}'.format(f1(probs, targets, num_classes=3, average='macro')))
    print('Micro F-1 - {:.4f}'.format(f1(probs, targets, num_classes=3, average='micro')))
    
    precisions, recalls, thresholds = precision_recall_curve(probs, targets, num_classes=3)
    average_precisions = average_precision(probs, targets, num_classes=3)
    
    pr_df = pd.concat([
        pd.DataFrame({
            'Precision':precisions[i].cpu(),
            'Recall':recalls[i].cpu(),
            'average_precision':average_precisions[i].cpu().item(),
            'area': auc(recalls[i], precisions[i]).cpu().item(),
            'class':classes[i]
        }) for i in range(3)],ignore_index=True)
    
    pr_df.to_csv('pr_df2.csv', index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
        
    fprs, tprs, thresholds = roc(probs, targets, num_classes=3)
    roc_df = pd.concat([
        pd.DataFrame({
            'fpr':fprs[i].cpu(),
            'tprs':tprs[i].cpu(),
            'area':auc(fprs[i], tprs[i]).cpu().item(),
            'class':classes[i]
        }) for i in range(3)],ignore_index=True)
    
    roc_df.to_csv('roc_df2.csv', index=False)
    
    
    
    cm = confusion_matrix(probs, targets, num_classes=3).cpu().numpy().astype(int)
    np.save('cm_array2', cm)
    
            
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument("-c","--use-cuda", action="store_true", help="try to use GPU")
    parser.add_argument("-a", "--arch", type=str, default="res", choices=["res", "conv"], help="architecture of pretrained model")
    parser.add_argument("-s", "--state-dict", type=str, default="trained_model.pt", help="pretrained model filename")
    parser.add_argument("-b","--batch-size", type=int, default=32, help="test dataloader batch size")
    parser.add_argument("-p", "--memory-pinning", action="store_true", help="copy tensors to CUDA pinned memory")
    parser.add_argument("-w","--workers",type=int, default=4, help="number of subprocesses for data loading")
    args = parser.parse_args()
    main(args)
    
        