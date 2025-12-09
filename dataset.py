import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

from deepaction_extract import Video, Model_Data
import config


'''
This module provides PyTorch components for video classification.

ClipData is a PyTorch Dataset class that loads video frames and applies transforms.
it takes a list of Video objects and converts them into tensors suitable for training.
frames are loaded on-demand when accessed through the DataLoader.

CNN is a convolutional neural network model for video classification.
it processes video frames through convolutional layers to extract spatial features,
then averages features across time and uses fully connected layers for classification.

''' 

class ClipData(Dataset):
    # clips: list of Video objects
    # transform: image transforms to apply
    # frames: number of frames to sample per video
    def __init__(self, clips, transform=None, frames=config.NUM_FRAMES):
        self.paths  = [c.video_path for c in clips]
        self.labels = [c.label_index for c in clips]
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    # i: index of the video to load
    def __getitem__(self, i):
        path  = self.paths[i]
        label = self.labels[i]

        # using the Video class from deepaction_extract.py 
        temp_vid = Video(path, label, "temp", sample_size=self.frames)  # temporary Video object to load frames
        bgr_frames = temp_vid.selected_frames  # list of frames in BGR format (numpy arrays)

        tensors = []
        for frame in bgr_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            if self.transform:
                t = self.transform(pil_img)
            else:
                t = transforms.ToTensor()(pil_img)
            tensors.append(t)

        return torch.stack(tensors), label



class CNN(nn.Module):
    # n_classes: number of video classes to classify
    # n_frames: number of frames per video
    def __init__(self, n_classes, n_frames=config.NUM_FRAMES):
        super().__init__()          
        #conv_stack is the convolutional stack of the model
        #convolutional layers are used to extract features from the video frames
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True), nn.MaxPool2d(2),                                                   
        )

        # after three 2×2 pools: calculate final spatial size dynamically
        # config.IMG_SIZE -> config.IMG_SIZE/2 -> config.IMG_SIZE/4 -> config.IMG_SIZE/8
        final_spatial = config.IMG_SIZE // 8
        self.head = nn.Sequential(
            nn.Linear(128 * final_spatial * final_spatial, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),           nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, n_classes),     # raw logits
        )

    # x: input tensor(batch, time, channels, height, width)
    def forward(self, x):
        B, T, C, H, W = x.shape  # B: batch size, T: num frames, C: channels, H: height, W: width ### 1, 10, 3, 128, 128 
        x = x.view(B * T, C, H, W)
        x = self.conv_stack(x)
        x = x.view(B, T, -1)
        x = x.mean(dim=1)           # temporal average
        return self.head(x)

"""
class RNN(nn.Module):
    # n_classes: number of video classes to classify
    # n_frames: number of frames per video
    def __init__(self, n_classes, n_frames=NUM_FRAMES):
"""