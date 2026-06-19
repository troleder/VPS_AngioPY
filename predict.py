import argparse
import logging
import os

import torch
# Limit PyTorch to a single thread to prevent CPU pinning (which freezes Streamlit heartbeats)
# and to avoid high memory/CPU usage on shared VPS environments.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from utils.dataset import CoronaryDataset
import segmentation_models_pytorch.segmentation_models_pytorch as smp

from torch.backends import cudnn

'''
This uses a pytorch coronary segmentation model (EfficientNetPLusPlus) that has been trained using a freely available dataset of labelled coronary angiograms from: http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html
The input is a raw angiogram image, and the output is a segmentation mask of all the arteries. This output will be used as the 'first guess' to speed up artery annotation. 
'''

def predict_img(net, dataset_class, full_img, device, scale_factor=1, n_classes=3):
    # NOTE n_classes is the number of possible values that can be predicted for a given pixel. In a standard binary segmentation task, this will be 2 i.e. black or white

    net.eval()

    img = torch.from_numpy(dataset_class.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if n_classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        full_mask = tf(probs.cpu())   

    if n_classes > 1:
        return dataset_class.one_hot2mask(full_mask)
    else:
        return full_mask > 0.5


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-d', '--dataset', type=str, help='Specifies the dataset to be used', dest='dataset', required=True)
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')

    return parser.parse_args()


