import torch
import torch.nn as nn
import sys
from math import floor 
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.attention import SpatialTransformer
from modules.pos_emb import PosEmbeds
import os
import numpy as np
import matplotlib.pyplot as plt
from casadi import vertcat , DM
import l4casadi as l4c
import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"
class Autoencoder_path(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        hidden_channels=32,
        attn_blocks=4,
        attn_heads=4,
        cnn_dropout=0.15,
        attn_dropout=0.15,
        downsample_steps=3,
        resolution=(256, 256),
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

        self.encoder_after = Encoder(32, 32, 1, 0.15, num_groups=32)
        self.decoder_after = Decoder(32, 32, 1, 0.15, num_groups=32)
        self.linear_after_max = nn.Sequential(
            nn.Linear(4416, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.linear_after_mean = nn.Sequential(
            nn.Linear(4416, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.recon_criterion = nn.MSELoss()
        self.mode = mode
        self.k = 1
        self.automatic_optimization = False
        self.device = torch.device("cuda")
     #   self.save_hyperparameters()

    def forward(self, batch):
        batch = batch.reshape(-1, 4355)

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
        encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean


    def encode_map_footprint(self, batch):
        mapp = batch[..., :65536].to(self.device)
        mapp = torch.reshape(mapp, (-1, 1, 256, 256))

        footprint = batch[..., 65536:].to(self.device)
        footprint = torch.reshape(footprint, (-1, 1, 256, 256))

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
        batch = batch.reshape(-1, 4355)

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
        encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_max, encoded_input_mean

    def step_ctrl(self, batch):
        mapp, x_crd, y_crd, theta = batch
        map_encode = self.encoder(mapp[:, :1, :, :])

        map_encode_robot = (
            self.encoder_robot(mapp[:, -1:, :, :]).flatten().view(mapp.shape[0], -1)
        )
        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = map_encode  # torch.cat((map_encode, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1)
        encoded_input = self.encoder_after(encoded_input)
        encoded_input = self.decoder_after(encoded_input)
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0], -1)

        encoded_input = torch.cat(
            (
                encoded_input,
                map_encode_robot,
                x_cr_encode,
                y_cr_encode,
                tsin_encode,
                tcos_encode,
            ),
            1,
        )

        encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_max, encoded_input_mean

    def training_step(self, batch, output):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        loss = self.step_ctrl(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss

    def step(self, batch, batch_idx, regime):
        map_design, start, goal, gt_hmap = batch
        inputs = (
            torch.cat([map_design, start + goal], dim=1)
            if self.mode in ("f", "nastar")
            else torch.cat([map_design, goal], dim=1)
        )
        predictions = self(inputs)

        loss = self.recon_criterion((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f"{regime}_recon_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        loss = self.step(batch, batch_idx, "train")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-4, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]


def test_model(model_loaded , num_map , d_map , d_footprint):

    ####### Test L4Casadi model
    l4c_model = l4c.L4CasADi(model_loaded, model_expects_batch_dim=True , name='y_expr',device='cuda')

    print("l4model " , l4c_model)

    map_inp_test = torch.zeros((131072))

    
    # print(len(input_model))
    k = 0
    for i in range (256):
        for j in range(256):
            if(num_map==-1):
                map_inp_test[k] = d_map['sub_maps'][i,j]
            else:
                map_inp_test[k] = d_map['sub_maps'][num_map][i,j]
           # map_inp_test[k] = data_sub_maps_1000['sub_maps'][num_map][i,j]
            if(map_inp_test[k]==10):
                map_inp_test[k] = 0
            k = k+1
    k = 0
    for i in range (256):
        for j in range(256):
            map_inp_test[65536+k] = d_footprint['footprint_husky'][i,j]
            k = k +1
    result = model_loaded.encode_map_footprint(map_inp_test).detach()
    print("result shape ", result.shape)
    input_model = DM(4288+32+32,1)
    for i in range(4352):
        input_model[i,0] = result[0,i].cpu().data.numpy()

    print("l4casadi model " , l4c_model(vertcat(input_model , 3.5 , 2.5 , 1.57)))


    resolution = 128
    res_array = np.zeros((resolution,resolution))
    k1 = 0
    k2 = 0
    
    for i in np.arange(0., 5.12, 5.12/resolution):
        k2 = 0
        for j in np.arange(0., 5.12, 5.12/resolution):
            res_array[k1 , k2] = l4c_model(vertcat(input_model , i , j , 1.5))
            k2 = k2 + 1
        k1 = k1 + 1
  #  for i in range(128):
  #      print(res_array[i,:])
    plt.imshow(np.rot90(res_array)+d_map['sub_maps'][num_map][::2,::2]*0.02)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda")
    model_loaded = Autoencoder_path(mode="k")
    model_loaded.to(device)

    load_check = torch.load("1000maps_zones_husky.pth")
    # model_dict = model_loaded.state_dict()
    # pretrained_dict = {k: v for k, v in load_check.items() if k in model_dict}
    # model_dict.update(pretrained_dict) 
    model_loaded.load_state_dict(load_check)
    model_loaded.eval();

    data_sub_maps_1000 = pickle.load( open( "dataset_sub_maps.pkl", "rb" ))
    data_footprints = pickle.load( open( "dataset_footprints.pkl", "rb" ))

    num_map = 0


    test_model(model_loaded , num_map , data_sub_maps_1000 , data_footprints)
