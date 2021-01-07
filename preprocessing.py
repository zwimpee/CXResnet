###############################################################################
# CXResnet project data preprocessing
#
# Author: Zachary Wimpee
#
# This file contains the code for downloading Curated X-Ray Dataset from Kaggle,
# getting the train/validation/testing splits, and calculating the values needed
# for normalization from the training data. Options are configurable via command
# line arguments from the user upon execution. 
#
#
# NOTE - Original dataset has separate classes for viral and bacterial pneumonia.
#        This distinction has been removed for the sake of model performance.
#        Future work on this project should attempt to reintroduce this
#        distinction and modify the network architecture accordingly.
###############################################################################

import argparse
from pathlib import Path
import json
import shutil

import pandas as pd
import numpy as np

from kaggle.api.kaggle_api_extended import KaggleApi


from sklearn.model_selection import train_test_split


import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms




def main(args):
    """
    This function is the main body for the file, but the functions
    defined outside of it can be imported and used independently in
    other files.
    """
    
    # Clear the destination directory if it already exists.
    if Path(args.dst_dir).is_dir():
        rmtree(Path(args.dst_dir))
    
    # Create list of relative directory paths.
    splits = ['train','test','val']
    labels = ['COVID','Normal','Pneumonia']
    dst_paths = ['/'.join([args.dst_dir,split,label]) for label in labels for split in splits]
    
    if args.download:
        if Path(args.src_dir).is_dir():
            rmtree(Path(args.src_dir))
        download_dataset(args)
    
    # Create the parent directory and subfolders.
    for dst_path in dst_paths:
        Path(dst_path).mkdir(parents=True)
        
    
    df = pd.DataFrame([(str(img_path),img_path.parent.stem.split('-')[0]) 
                       for img_path in sorted(Path(args.src_dir).glob('**/*.jpg'))],columns=['path','label'])
    
    # Get training, validation, and testing DataFrames.
    train_df, test_df = train_test_split(df, test_size = args.test_split, stratify = df['label'])
    train_df, val_df = train_test_split(train_df, test_size = args.val_split, stratify = train_df['label'])

    # Reset the DataFrame indices.
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df['path'] = train_df.apply(copy_images,split_path=(Path(args.dst_dir)/'train').resolve(),axis=1)
    val_df['path'] = val_df.apply(copy_images,split_path=(Path(args.dst_dir)/'val').resolve(),axis=1)
    test_df['path'] = test_df.apply(copy_images,split_path=(Path(args.dst_dir)/'test').resolve(),axis=1)
    
    # Remove the source directory.
    rmtree(Path(args.src_dir))

    # Add column to specify the split.
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Concat the DataFrames and write to csv file.
    df = pd.concat([train_df,val_df,test_df],ignore_index=True)
    df.to_csv('data_df.csv',index=False)
    
    mean_std = mean_std_calcs(args)
    
    with open('mean_std.json', 'w') as f:
        json.dump(mean_std, f)
        
    with open('mean_std.json') as f:
        r = json.load(f)
        # Print the dictionary values to check if it worked.
        print('Channel means - {}'.format(r['mean']))
        print('Channel std. devs. - {}'.format(r['std']))
        print('-'*10)
        
    print('preprocessing complete')




def rmtree(root):
    "Clear a directory and remove it."
    if not root.is_dir():
        print('Folder or directory does not exist.')
        return 
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()
    root.rmdir()
    
    

def download_dataset(args):
    "Download kaggle dataset via its API."
    # Initialize the api object.
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset zip file.
    api.dataset_download_cli(args.dataset, force=args.kforce, unzip=args.unzip)
    
    
def copy_images(x, split_path):
    """
    When applied to rows of a pandas DataFrame object the files in 
    the 'path' column are copied and moved to a folder associated with
    the 'label' column.
    """
    # Get source image path.
    source = x.path
    
    # Set destination directory.
    dest = (split_path/x.label).resolve()
    
    # Create a copy in the destination directory.
    return shutil.copy(source,dest)


def mean_std_calcs(args):
    "Calculate channel mean and standard deviations for a set of image files."
    
    # Get path to training folder.
    path = (Path(args.dst_dir)/'train').resolve()
    
    # Load dataset with ImageFolder.
    dataset = ImageFolder(path, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
    
    # Get num_samples random indices.
    indices = torch.randperm(len(dataset))[:args.n_samples]
    
    # Calculate channel means and stds.
    means = torch.mean(torch.stack([torch.mean(dataset[i][0], (1,2)) for i in indices]),0)
    stds = torch.mean(torch.stack([torch.std(dataset[i][0], (1,2)) for i in indices]),0)
    
    del dataset
    
    # Store tensors as lists in dict and write to json file.
    mean_std = {'mean': means.tolist(),'std': stds.tolist()}
    return mean_std
    
        
        

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset creation for Chest X-Ray image classification project.")
    parser.add_argument("-u","--dataset", type=str, default="unaissait/curated-chest-xray-image-dataset-for-covid19", help="dataset url suffix")
    parser.add_argument("--src-dir", type=str, default="Curated X-Ray Dataset/", help="path to source dataset to be split")
    parser.add_argument("--dst-dir", type=str, default="data/", help="path to destination parent directory for train/validation/testing splits")
    parser.add_argument("-t","--test-split", type=float, default=0.2, help="float value for fraction of total data to use for test split")
    parser.add_argument("-v","--val-split", type=float, default=0.25,help="float value for fraction of remaining train data to use for validation split")
    parser.add_argument('-n',"--n-samples",  type=int,default=1000,help="number of training samples for calculating normalization constants")
    parser.add_argument("-d","--download", action="store_true",help="download the original dataset")
    parser.add_argument("-f","--kforce", type=bool,default=False,help="choice to download if original dataset already exists")
    parser.add_argument("-z","--unzip", type=bool,default=True,help="choice unzip downloaded dataset")
    args = parser.parse_args()
    main(args)