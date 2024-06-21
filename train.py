from PIL import Image
import numpy as np
import torch.nn as nn
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
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import TensorDataset, Dataset, DataLoader
import os
import net
from train_GAN import Generator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# FaceResNet is just resnet with an extra last few layers for a new loss function which could be changed by self.head
class FaceResNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity() 

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(in_features=2048, out_features=512)
        self.dropout = nn.Dropout(p=0.4)
        self.cosine_layer = nn.Linear(512, 1, False)

        #change head for different loss functions
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

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    if np_img.any():
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        #tensor=tensor.to(device)
        #print(tensor)
        return torch.squeeze(tensor)
    else:
        return [0, 0]    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    gen_file = "generator_"+model_dir
    gen_v_file= "V_"+model_dir

    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    model_path = os.path.join(script_dir, 'models', model_dir)
    gen_path = os.path.join(script_dir, 'generators', gen_file)
    gen_v_path = os.path.join(script_dir, 'generators', gen_v_file)
    #open dataset
    with open(dataset_path, "rb") as fp:
        trainface_dat = pickle.load(fp)
    fp.close()
    
    #transform data into the right form and split into train and test data
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


    X_train, X_test, y_train, y_test = train_test_split(new_im, t_list, test_size=0.2, random_state=42)
    train_length=len(X_train)
    X_train_tensor = torch.stack(X_train)

    y_train_tensor = torch.tensor(y_train)
    new_im_tens=torch.stack(new_im)
    t_tens=torch.tensor(t_list)
    fulldat=TensorDataset(new_im_tens, t_tens)


    X_test_tensor = torch.stack(X_test)
    y_test_tensor = torch.tensor(y_test)
    batch_size=128
    uni=torch.unique(torch.tensor(t_tens, dtype=torch.long))
    num_classes=uni.size(dim=0)

    # Oversample the low number classes in the training set
    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    def check_data(data):
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError("Data contains NaN or infinity values")


    dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
    class_counts = torch.bincount(t_tens)
    few_shot_classes=[]
    for c in range(len(class_counts)):
        if class_counts[c]<10:
            few_shot_classes.append(c)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[i_encoded]
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(fulldat, batch_size=batch_size)
    test_laoder=DataLoader(dataset_test, batch_size=batch_size)
    num_batches = len(train_loader)
    total_samples = batch_size*num_batches
    test_length=len(test_laoder)*batch_size
    #initialize model
    #modelo = FaceResNet(num_classes).to(device)
    modelo =load_model(pretrained=True).to(device)
    #modelo.head = Normal(embedding_size=512, classnum=num_classes).to(device)
    #modelo.head = CosFace(embedding_size=512, classnum=num_classes, m=0.6,s=64).to(device)
    #modelo.head = ArcFace(embedding_size=512, classnum=num_classes, s=64., m=0.5).to(device)
    #modelo.head = CenterLoss(embedding_size=512, classnum=num_classes,alpha=0.6).to(device)
    modelo.head = AdaFace(embedding_size=512, classnum=num_classes, m=0.4, h=0.333, s=64, t_alpha=1.0).to(device)


    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer_ft = optim.Adam(modelo.parameters(), lr=0.0004, weight_decay=1e-4)
    latent_dim=100
    
    gen=True


    clust_im=True
    if gen==True:
        generator=torch.load(gen_path)
        with open(gen_v_path, "rb") as fp:
            V = pickle.load(fp)

    for epoch in range(5):  # loop over the dataset multiple times
        individual_losses = []
        cor=0
        running_loss = 0.0
        modelo.train()
        i=0
        gen_embeds=[]
        gen_labels=[]
        for rgb_batch, labels in train_loader:
            i+=1
            rgb_batch, labels = rgb_batch.to(device), labels.to(device)
            optimizer_ft.zero_grad()
            if gen==True:
                # Generate additional samples for few-shot classes
                num_generated_samples_per_class = 8 # Number of samples to generate per few-shot class
                generated_feats = []
                generated_labels = []
                generated_norms = []
                for lab in labels:
                    if lab in few_shot_classes: 
                        z = torch.randn(num_generated_samples_per_class, latent_dim).to(device)
                        cls_labels = torch.full((num_generated_samples_per_class,), lab, dtype=torch.long).to(device)
                        gen_images = generator(z, cls_labels,V)
                        norms = torch.norm(gen_images, 2, 1, True)
                        generated_norms.append(norms)
                        generated_feats.append(gen_images)
                        generated_labels.append(cls_labels)
            

                generated_feats = torch.cat(generated_feats)
                generated_labels = torch.cat(generated_labels)
                generated_norms=torch.cat(generated_norms)

            norms,embeddings=modelo.get_embed(rgb_batch.float())
            if gen==True:
                embeddings = torch.cat((embeddings, generated_feats))
                labels = torch.cat((labels, generated_labels))
                norms=torch.cat((norms,generated_norms))
            if clust_im==True and epoch>1:
                for integ in range(embeddings.size(dim=0)):
                    e=embeddings[integ]
                    la=labels[integ]
                    gen_embeds.append(e.clone().cpu().detach().numpy())
                    gen_labels.append(la.clone().cpu().detach().numpy())
            outputs = modelo.only_feats(embeddings,norms, labels)
            predicted=torch.argmax(outputs ,dim=1)
            cor += (predicted == labels).sum().item() 
            losses = criterion(outputs,labels)
            for j in losses:
                individual_losses.append(j.item())
            loss = losses.mean()
            if type(modelo.head) ==CenterLoss:
                c_loss=modelo.head.center_loss(embeddings,labels)
                loss+=c_loss
                

            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item()
        mean_loss = np.mean(individual_losses)
        # test batches
        modelo.eval()
        test_cor=0
        with torch.no_grad():
            for rgb_batch_test, labels_test in test_laoder:
                norms_test,embeddings_test=modelo.get_embed(rgb_batch_test.to(device).float())

                outputs_test= modelo.only_feats(embeddings_test,norms_test, labels_test)
                predicted_test = torch.argmax(outputs_test, dim=1)

                test_cor += (predicted_test == labels_test).sum().item()
        
        print(test_cor)
        print(test_length)
        print(f"test_accuracy = {test_cor /test_length}")
        print(f'[{epoch + 1}] loss: {running_loss / total_samples:.3f}')
        print(f"correct = {cor}")
        print(f"accuracy = {cor /total_samples}")
        if cor /total_samples>0.95:
            break
    if clust_im==True:
        # Generate a list of colors using a seaborn color palette
        unique_labels = np.unique(gen_labels)
        n_colors=len(unique_labels)
        palette = sns.color_palette("Spectral", n_colors)  # "hsv" is a good choice for distinct colors

        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        pca_2=PCA(2)
        tot_feats=np.array(gen_embeds)
        #Transform the data
        df = pca_2.fit_transform(tot_feats)
        for lab in unique_labels:
            indices = np.where(gen_labels == lab)
            plt.scatter(df[indices, 0], df[indices, 1], color=label_to_color[lab], label=lab, s=10, alpha=0.6)
        #for p in range(len(df)):
            #point=df[p]
            #lab=gen_labels[p]
           #plt.scatter(point[0],point[1],label=lab)
        plt.show()
    torch.save(modelo, model_dir)

