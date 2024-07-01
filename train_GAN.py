import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import torchvision.transforms as transforms
import random
import net
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torchvision.models as models
from collections import defaultdict
import torch.optim as optim
from sklearn.decomposition import PCA
import numpy as np
import os
import sys

# Define the class of the regular model
class FaceResNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity() 

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(in_features=2048, out_features=512)
        self.dropout = nn.Dropout(p=0.4)
        self.cosine_layer = nn.Linear(512, 1, False)

    def forward(self, x, labels):
        
        x = self.model(x)
        
        norms = torch.norm(x, 2, 1, True)
        features = torch.div(x, norms)
        features = self.dropout(features)

        if type(self.head) == CenterLoss or type(self.head)==Normal:
            logits = self.head(features, norms)
        else:
            logits = self.head(features, norms, labels)
        return features, logits
    def get_embed(self, x):
        x = self.model(x)
        norms = torch.norm(x, 2, 1, True)
        features = torch.div(x, norms)
        return norms,features
    
    def cosine(self,x):
        x = self.get_embed(x)
        x = nn.functional.normalize(x, p=2.0, dim=1)
        x = self.cosine_layer(x)

        return x
    def only_feats(self, features,norms, labels):
        logits = self.head(features,norms,labels)
        return logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, feature_dim)
        
        self.fc1 = nn.Linear(feature_dim + feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc4 = nn.Linear(1024, feature_dim)
    
    def forward(self, noise, labels,V):
        projected_noise = torch.matmul(noise, V.T)
        label_embedding = self.label_emb(labels)
        gen_inp = torch.cat((projected_noise, label_embedding), -1)

        x = F.relu(self.bn1(self.fc1(gen_inp)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc1 = nn.Linear(feature_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, features, labels):
        label_embedding = self.label_emb(labels)
        disc_input = torch.cat((features, label_embedding), -1)
        x = F.leaky_relu(self.fc1(disc_input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return torch.sigmoid(x)
    
if __name__ == "__main__":
    # Get the right data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    gen_file = "generator_"+model_dir
    gen_v_file= "V_"+model_dir

    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    model_path = os.path.join(script_dir, 'models', model_dir)
    gen_path = os.path.join(script_dir, 'generators', gen_file)
    gen_v_path = os.path.join(script_dir, 'generators', gen_v_file)
    with open(dataset_file, "rb") as fp:
        trainface_dat = pickle.load(fp)
        fp.close()
        
    #transform data into the right form for input
    image_list_uns=trainface_dat[0]
    target_list_uns=trainface_dat[1]
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    new_t=[]
    new_im=[]
    temp = list(zip(image_list_uns, target_list_uns))
    random.shuffle(temp)
    image_list, target_list = zip(*temp)
    num_face=len(image_list)




    for i in range(num_face):
        im=Image.fromarray(image_list[i]).convert('RGB')
        new_t.append(target_list[i])
        imag=transform(im).to(device)
        new_im.append(imag)
    l_encoder = LabelEncoder()
    i_encoded = l_encoder.fit_transform(new_t)
    t_list=torch.from_numpy(i_encoded).to(device)
    new_im_tens=torch.stack(new_im)
    t_tens=torch.tensor(t_list)
    noise_dim = 100
    feature_dim = 512  # Example feature dimension, depends on the pre-trained model's output
    uni=torch.unique(torch.tensor(t_tens, dtype=torch.long))
    num_classes=uni.size(dim=0)


    fulldat=TensorDataset(new_im_tens, t_tens)
    batch_size=60
    train_loader = DataLoader(fulldat, batch_size=batch_size,drop_last=True)
    generator = Generator(noise_dim, feature_dim, num_classes).to(device)
    discriminator = Discriminator(feature_dim, num_classes).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    num_epochs=600
    modelo=torch.load(model_path)

    # Calculate average variance per class
    class_means = np.zeros((num_classes, 512))
    class_counts = np.zeros((num_classes))
    with torch.no_grad():
        for rgb_batch, labels in train_loader:
            _, features = modelo.get_embed(rgb_batch.float())
            features = features.cpu().numpy()
            labels = labels.cpu().numpy()

            for label, feature in zip(labels, features):
                class_means[label] += feature
                class_counts[label] += 1
        for i in range(len(class_means)):
            class_means[i] /= class_counts[i]
        centered_features=[]
        for rgb_batch, labels in train_loader:
            _, features = modelo.get_embed(rgb_batch.float())
            features = features.cpu().numpy()
            labels= labels.cpu().numpy()

            for feature, label in zip(features, labels):
                centered_features.append(feature - class_means[label])

    # Calculate principal components with PCA
    n_components = 100  # Number of principal components to keep
    pca = PCA(n_components=n_components)
    pca.fit(np.array(centered_features)) 

    # Define the projection matrix
    V = pca.components_.T  # Shape: (n_features, n_components)
    V = torch.tensor(V, dtype=torch.float32).to(device)


    # Train the GAN
    for epoch in range(num_epochs):
        for rgb_batch, labels in train_loader:
            _,features=modelo.get_embed(rgb_batch.float())
            real_features = features.to(device)
            real_labels = labels.to(device)
            real_targets = torch.ones(batch_size, 1).to(device)

            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_features = generator(noise, fake_labels, V)
            fake_targets = torch.zeros(batch_size, 1).to(device)

            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_features, real_labels), real_targets)
            fake_loss = criterion(discriminator(fake_features.detach(), fake_labels), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            gen_loss = criterion(discriminator(fake_features, fake_labels), real_targets)
            gen_loss.backward()
            optimizer_G.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")

    torch.save(generator, gen_path)
    with open(gen_v_file, 'wb') as fp:
        pickle.dump(V, fp)

