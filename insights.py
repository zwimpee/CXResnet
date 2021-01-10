###############################################################################
# CXResnet trained model prediction insights
#
# Author: Zachary Wimpee
#
# Decent performance metric values for the trained model are promising, but 
# this alone does not provide much useful insight, nor does it yield anything 
# of signficant value with respect to a deliverable product. Therefore here we
# use the Captum library in an attempt to gain insight into what image features
# are influencing the model's predictions, which hopefully will yield some level
# of interpretability.
#
# NOTE - File is currently only compatible with the Occlusion attribution algorithm
#        in order to allow for fine-tuned control of the algorithm parameters.
#        This limited functionality will be expanded in future work.
# 
#
###############################################################################
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from architecture import ResidualNet, ConvNet
import matplotlib.pyplot as plt
from PIL import Image
import random
from ast import literal_eval

from captum.attr import Occlusion
from captum.attr import visualization as viz

from testing import get_model
from preprocessing import rmtree

# Define tuple for class clabels. 
classes = ('COVID', 'Normal', 'Pneumonia')

# Read training set channel mean and std from json file.
with open('mean_std.json') as f:
    mean_std = json.load(f)

mean = mean_std['mean']
std = mean_std['std']

# Define transforms.
# Transforms are split differently than 
# other files in order to control how they are applied.
resize = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])

normalize = transforms.Compose([
    transforms.ToTensor(), # converts image to tensor
    transforms.Normalize(mean,std) # normalize the channel values
])

ToTensor = transforms.ToTensor() # used to get the original unnormalized image

def get_image_paths(n_images=3):
    "Returns a list of random test set image paths"
    
    # Get the list of test set image paths.
    image_list = sorted(Path('data/test/').glob('**/*.jpg'))
    
    # Select n_imgs random images.
    image_paths = random.sample(image_list, n_images)
    
    return image_paths

def get_image_data(image_path):
    "Returns the original and transformed image in addition to its class label index."
    
    # Open the image with PIL.Image.
    image = Image.open(image_path)
    
    # Get the class label index.
    label_idx = classes.index(image_path.parent.stem)
    
    # Get transformed image.
    input_image = normalize(resize(image))
    
    # Get numpy array for resized original image.
    original_image = np.transpose(ToTensor(resize(image)).numpy(), (1,2,0))
    
    return original_image, input_image, label_idx


def get_attribution(occlusion, input_image, target, args):
    """
    Use Occlusion to get the attribution of the input to the model prediction.
    
    Args:
        occlusion - Attribution interpreter instance.
        
        input_image = Transformed image model input.
        
        target - Class index to use for input attribution. 
                         
        args - Additional arguments from command line execution specifying the parameters
               for the Occlusion attribution.
    """
    
    # Add batch dimension to input image and track gradients.
    input_image = input_image.unsqueeze(0)
    input_image.requires_grad = True
    
    # Define arguments for attribution.
    target = target.item(),    
    baselines = args.baselines
    strides = literal_eval(args.strides)
    sliding_window_shapes = literal_eval(args.sliding)
    
    # Compute and return the attribution of the input to the target index.
    attribution_target = occlusion.attribute(input_image,strides=strides,target=target,sliding_window_shapes=sliding_window_shapes,baselines=baselines)
    attribution_target = np.transpose(attribution_target.squeeze().cpu().detach().numpy(), (1,2,0))
    
    return attribution_target

def plot_attribution(model, occlusion, path, fig, ax, args):
    "Plot the occlusion attribution for image at given path."
    # Get data for the image found at input path.
    original_image, input_image, label = get_image_data(path)
    
    # Ensure model is in evaluation mode.
    model.eval()
    with torch.no_grad():
        pred_score, pred_label_idx = torch.max(F.softmax(model(input_image.unsqueeze(0)),1),1)
        pred_label = classes[pred_label_idx.item()]
        
    attribution_target = get_attribution(occlusion, input_image, pred_label_idx, args)
    ax.set_title(
        "Predicted label: {0}, {1:.1f}%\n True label: {2}"
        .format(
            pred_label,
            pred_score.squeeze().item()*100.0,
            classes[label]
        ),color=("green" if pred_label==classes[label] else "red"))
    return viz.visualize_image_attr(attribution_target,original_image,"blended_heat_map","all",plt_fig_axis=(fig, ax),show_colorbar=True,alpha_overlay=args.alpha_overlay,use_pyplot=False)
    


def main(args):
    # Assign device specified by user.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    if args.clear_figures:
        rmtree(Path('figures'))
        Path('figures').mkdir()
    
    # Get pretrained model.
    model = get_model(args.arch,args.state_dict,device)
    model.eval()
    
    # Define the occlusion interpreter.
    occlusion = Occlusion(model)
    
    # Get paths to test images.
    image_paths = get_image_paths(args.n_images)
    
    # Save and show attributions individually by default.
    if not args.subplots:
        for i in range(args.n_images):
            fig, ax = plt.subplots(figsize=(args.figsize,args.figsize),subplot_kw=dict(xticks=[],yticks=[]))
            fig, ax = plot_attribution(model, occlusion, image_paths[i], fig, ax, args)
            filename = args.save_image+'_{}.png'.format(image_paths[i].stem.replace(" ", "_"))
            plt.savefig("/".join(["figures",filename]), facecolor=args.facecolor)
            plt.show()
            if args.tb_write:
                run = '/'.join(['runs', args.tb_folder])
                writer = SummaryWriter(run)
                writer.add_figure(filename.split('.')[0], fig)
                writer.close()
            plt.close()
    else:
        # Create the subplot figure.
        fig, axs = plt.subplots(1,args.n_images,
                                figsize=(args.figsize*args.n_images,args.figsize),
                                subplot_kw=dict(xticks=[],yticks=[]),
                                tight_layout=True)
        for i in range(args.n_images):
            _ = plot_attribution(model, occlusion, image_paths[i], fig, axs[i], args)
            
        plt.tight_layout()
        filename = args.save_image+"_multi.png"
        plt.savefig("/".join(["figures",filename]), facecolor=args.facecolor)
        plt.show()
        if args.tb_write:
            run = '/'.join(['runs', args.tb_folder])
            writer = SummaryWriter(run)
            writer.add_figure(filename.split('.')[0], fig)
            writer.close()
        plt.close()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model insights and interpretation")
    parser.add_argument("-c","--use-cuda", action="store_true", help="try to use GPU")
    parser.add_argument("-x","--clear-figures", action="store_true", help="clear all images in figures folder")
    parser.add_argument("-a", "--arch", type=str, default="res", choices=["res", "conv"], help="architecture of pretrained model")
    parser.add_argument("-m", "--state-dict", type=str, default="trained_model.pt", help="pretrained model weights filename")
    parser.add_argument("-n","--n-images", type=int, default=1, help="number of test images to evaluate")
    parser.add_argument("-y", "--save-image", type=str,default="occlusion_attribution", help="filename for saving image")
    parser.add_argument("-p", "--figsize",type=int, default=8, help="subplot figure dimension")
    parser.add_argument("--subplots", action="store_true", help="plot all input attributions on single figure")
    parser.add_argument("-o", "--alpha-overlay",type=float, default=0.5, help="alpha value for heatmap overlay")
    parser.add_argument("-g", "--facecolor", type=str, default="auto", help="background color for saved figure")
    parser.add_argument("-b", "--baselines", type=int, default=0, help="value to replace occluded features")
    parser.add_argument("-s", "--strides", type=str, default="(3,9,9)", help="stride length for the occlusion sliding window")
    parser.add_argument("-w", "--sliding", type=str, default="(3,45,45)", help="occlusion sliding window shape")
    parser.add_argument("-t", "--tb-write", action="store_true", help="write the resultant figure to tensorboard log")
    parser.add_argument("-f", "--tb-folder", type=str, default="cxresnet", help="specify the tensorboard log folder")
    args = parser.parse_args()
    main(args)
    