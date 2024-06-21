from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU
import torch
from random import randrange
import torch.optim as optim
import pickle
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import torchvision.models as models
import random
from sklearn.model_selection import train_test_split
import sys, getopt
from head import AdaFace, CosFace, ArcFace, CenterLoss,Normal
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader
import os
import net
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def load_model(pretrained=False):
    # load model and pretrained statedict
    model = net.build_model()
    if pretrained==True:
        statedict = torch.load("X:\Downloads\AdaFace-master\AdaFace-master\pretrained/adaface_ir50_ms1mv2.ckpt")['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        model.load_state_dict(model_statedict)
    model.to(device)
    return model

class TripletDataset(Dataset):
    def __init__(self, anchor_images, positive_images, negative_images):
        self.anchor_images = anchor_images
        self.positive_images = positive_images
        self.negative_images = negative_images
        
    def __len__(self):
        return len(self.anchor_images)
    
    def __getitem__(self, idx):
        anchor_img = self.anchor_images[idx]
        positive_img = self.positive_images[idx]
        negative_img = self.negative_images[idx]
        
        return anchor_img, positive_img, negative_img

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute Euclidean distances between anchor, positive, and negative embeddings
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        # Compute triplet loss
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        
        return torch.mean(loss)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    model_path = os.path.join(script_dir, 'models', model_dir)

    #open dataset
    with open(dataset_path, "rb") as fp:
        tripface_d = pickle.load(fp)
    fp.close()

    tripface_dat=tripface_d[0]
    trip_anch_list_uns=tripface_dat[0]
    trip_pos_list_uns=tripface_dat[1]
    trip_neg_list_uns=tripface_dat[2]



    transform = transforms.Compose([
        transforms.PILToTensor()
    ])





        
    temp = list(zip(trip_anch_list_uns, trip_pos_list_uns,trip_neg_list_uns))
    random.shuffle(temp)
    trip_anch_list, trip_pos_list, trip_neg_list= zip(*temp)
    num_face=len(trip_anch_list)
    new_im_anch=[]
    new_im_pos=[]
    new_im_neg=[]
    for i in range(len(trip_anch_list)):
        im_anch=Image.fromarray(trip_anch_list[i]).convert('RGB')
        im_pos=Image.fromarray(trip_pos_list[i]).convert('RGB')
        im_neg=Image.fromarray(trip_neg_list[i]).convert('RGB')
        imag_anch=transform(im_anch).to(device)
        imag_pos=transform(im_pos).to(device)
        imag_neg=transform(im_neg).to(device)
        new_im_anch.append(imag_anch)
        new_im_pos.append(imag_pos)
        new_im_neg.append(imag_neg)




    new_im_anch_tens=torch.stack(new_im_anch)
    new_im_pos_tens=torch.stack(new_im_pos)
    new_im_neg_tens=torch.stack(new_im_neg)

    batch_size=40
    


    def check_data(data):
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError("Data contains NaN or infinity values")

    trips_dataset = TripletDataset(new_im_anch_tens, new_im_pos_tens, new_im_neg_tens)
    trips_dataloader = DataLoader(trips_dataset, batch_size=batch_size, shuffle=True)

        

    modelo =load_model(pretrained=True).to(device)
    optimizer_ft = optim.Adam(modelo.parameters(), lr=0.00001)
    print(len(new_im_anch_tens))
    for epoch in range(1):  # loop over the dataset multiple times
        individual_losses = []
        cor=0
        running_loss = 0.0
        modelo.train()
        i=0
        for batch in trips_dataloader:
            anchor_batch, positive_batch, negative_batch = batch
            _,anchor_embeddings = modelo.get_embed(anchor_batch.float())
            _,positive_embeddings = modelo.get_embed(positive_batch.float())
            _,negative_embeddings = modelo.get_embed(negative_batch.float())

            # Compute triplet loss
            criterion = TripletLoss(margin=0.2)
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            individual_losses.append(loss)
            loss.backward()
            optimizer_ft.step()
            i+=1
            running_loss += loss.item()
            print(batch_size*i, " / ", len(new_im_anch_tens))
            print(running_loss/i)
        print(running_loss /  len(new_im_anch_tens))
        # test batches
        test_cor=0
    torch.save(modelo, f"{sys.argv[1]}")

