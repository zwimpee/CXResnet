"""
Author: Zachary Wimpee

Description: Utility functions for model 
             training and testing.
             
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import architecture
import matplotlib.pyplot as plt
from PIL import Image
import random

from captum.attr import Occlusion
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

classes = ('COVID','Normal','Pneumonia')

# Read mean and std from json file.
with open('mean_std.json') as f:
    mean_std = json.load(f)

mean = mean_std['mean']
std = mean_std['std']

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
    outputs = model(images)
    
    # Get index of predicted class.
    preds = outputs.argmax(dim=1).tolist()
    
    # Get the probability of the predictions.
    probs = [F.softmax(outputs[i],dim=0)[preds[i]].item() for i in range(len(preds))]
    
    return preds, probs

def show_predictions(model, images, labels, mean=mean, std=std):
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
    
    
    
    