import sys
import os
import glob
import time
import argparse
import transformations as tf2
import tf
import pandas as pd
        
import os, json
from json import JSONEncoder
import numpy
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import rospy
import pytorch_lightning as pl
from std_msgs.msg import Float64MultiArray , Int32MultiArray
# import rosbag
# from cv_bridge import CvBridge
import message_filters
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# import cv2
import numpy as np
import math

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
# import keyboard
# import json
# from gym.spaces.box import Box
from collections import OrderedDict, defaultdict, deque
import torch.distributed as dist
import attr
import numbers
from  torch.cuda.amp import autocast
import pickle

sys.path.append('/home/vladislav/test_npfield/TransPath')
from modules.attention import SpatialTransformer
from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.pos_emb import PosEmbeds
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

my_msg = Float64MultiArray()
pub_map_emb = rospy.Publisher('map_footprint_emb', Float64MultiArray,queue_size=10)

d_footprint = pickle.load(open("dataset_footprint_husky.pkl", "rb"))
footprint = d_footprint['footprint_husky']
map_inp_1 = torch.zeros((5000))
img_footprint = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        img_footprint[i,j] = footprint[i][j]
k = 0
for i in range (50):
    for j in range(50):
        map_inp_1[2500+k] = img_footprint[i,j]
        if(map_inp_1[2500+k]==100):
            map_inp_1[2500+k] = 1
        k = k +1


class Autoencoder_path(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=8,
        hidden_channels=64,
        attn_blocks=4,
        attn_heads=4,
        cnn_dropout=0.15,
        attn_dropout=0.15,
        downsample_steps=3,
        resolution=(50, 50),
        mode="f",
        *args,
        **kwargs,
    ):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(
            1, hidden_channels, downsample_steps, cnn_dropout, num_groups=32
        )
        self.encoder_robot = Encoder(1, 1, 4, 0.15, num_groups=1)
        self.pos = PosEmbeds(
            hidden_channels,
            (
                resolution[0] // 2**downsample_steps,
                resolution[1] // 2**downsample_steps,
            ),
        )
        self.transformer = SpatialTransformer(
            hidden_channels, attn_heads, heads_dim, attn_blocks, attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels,
            (
                resolution[0] // 2**downsample_steps,
                resolution[1] // 2**downsample_steps,
            ),
        )
        self.decoder = Decoder(hidden_channels, out_channels, 1, cnn_dropout)

        self.x_cord = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.y_cord = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.theta_sin = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.theta_cos = nn.Sequential(nn.Linear(1, 16), nn.ReLU())

        self.encoder_after = Encoder(hidden_channels, 32, 1, 0.15, num_groups=32)
        self.decoder_after = Decoder(32, hidden_channels, 1, 0.15, num_groups=32)
        
        self.decoder_MAP = Decoder(hidden_channels, 2, 3, 0.15, num_groups=32)
        
        self.linear_after_mean = nn.Sequential(
            nn.Linear(1225, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.recon_criterion = nn.MSELoss()
        self.mode = mode
        self.k = 1
        self.automatic_optimization = False
        self.device = torch.device("cuda")
      #  self.save_hyperparameters()

    def forward(self, batch):
        batch = batch.reshape(-1, 1164)

        map_encode_robot = batch[..., :-3].to(self.device)

        x_crd = torch.reshape(batch[..., -3:-2].to(self.device), (-1, 1))
        y_crd = torch.reshape(batch[..., -2:-1].to(self.device), (-1, 1))
        theta = torch.reshape(batch[..., -1:].to(self.device), (-1, 1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat(
            (map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1
        )
        # encoded_input = self.linear_after(encoded_input)
        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean

    def encode_map_footprint(self, batch):
        mapp = batch[..., :2500].to(self.device)
        mapp = torch.reshape(mapp, (-1, 1, 50, 50))

        footprint = batch[..., 2500:].to(self.device)
        footprint = torch.reshape(footprint, (-1, 1, 50, 50))

        map_encode = self.encoder(mapp)

        map_encode_robot = (
            self.encoder_robot(footprint).flatten().view(mapp.shape[0], -1)
        )

        encoded_input = self.encoder_after(map_encode)
        encoded_input = self.decoder_after(encoded_input)
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0], -1)

        encoded_input = torch.cat((encoded_input, map_encode_robot), -1)

        return encoded_input

    def encode_map_pos(self, batch):
        batch = batch.reshape(-1, 1164)

        map_encode_robot = batch[..., :-3].to(self.device)

        x_crd = torch.reshape(batch[..., -3:-2].to(self.device), (-1, 1))
        y_crd = torch.reshape(batch[..., -2:-1].to(self.device), (-1, 1))
        theta = torch.reshape(batch[..., -1:].to(self.device), (-1, 1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat(
            (map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1
        )
        # encoded_input = self.linear_after(encoded_input)
        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean




device = torch.device("cuda")
model_loaded = Autoencoder_path(mode="k")
model_loaded.to(device)
load_check = torch.load("maps_50_MAP_LOSS.pth")
# model_dict = model_loaded.state_dict()
# pretrained_dict = {k: v for k, v in load_check.items() if k in model_dict}
# model_dict.update(pretrained_dict) 
model_loaded.load_state_dict(load_check)
model_loaded.eval();
losses = []
print("model is loaded")


def map_callback(msg):

    global map_inp_1
    local_map = msg.data

    img = np.zeros((50,50))
    k = 0
    for i in range(49,0,-1):
        for j in range(50):
            img[i,j] = local_map[k]
            k = k+1

    k = 0
    for i in range (50):
        for j in range(50):
            map_inp_1[k] = img[i,j]
            if(map_inp_1[k] == 100 or map_inp_1[k] == -1):
                map_inp_1[k] = 1
            k = k+1
    
    map_embedding = model_loaded.encode_map_footprint(map_inp_1).detach().cpu().data.numpy()
    my_msg.data = map_embedding[0,:].tolist()
    pub_map_emb.publish(my_msg)
   # print(my_msg.data)
    print("published embeding") 
    
    
       
    
def potential():
    rospy.init_node('map_potential_predictor', anonymous=True) 

    rospy.Subscriber('image', Int32MultiArray, map_callback) 

    rospy.spin()






def main():
    potential()

    print("Finish")


if __name__ == '__main__':
    main()          