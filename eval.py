import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from train import FaceResNet
import sys
import os

# Function to generate embeddings from two lists of images, a list of probes and a gallery
def get_features(model_path,image_path,orig):
    obj = pd.read_pickle(image_path)

    modelo=torch.load(model_path)
    modelo.eval()
    features = []
    features_y=[]
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    new_obj=[[],[]]
    images=[]
    #make embeddings of the two lists
    for i in range(len(obj[0])):
        

        im=Image.fromarray(obj[0][i]).convert('RGB')
        im_y=Image.fromarray(obj[1][i]).convert('RGB')
        images.append(im)
        x=transform(im).to(device)
        x=torch.unsqueeze(x, 0)

        y=transform(im_y).to(device)
        y=torch.unsqueeze(y, 0)
        new_obj[0].append(x)
        new_obj[1].append(y)
        if orig==True:
            _,feature = modelo.get_embed(to_input(im)[0].to(device).float())
            
            _,feature_y = modelo.get_embed(to_input(im_y)[0].to(device).float())
        else:
            _,feature = modelo.get_embed(x.float())
            
            _,feature_y = modelo.get_embed(y.float())            


        features.append(feature.cpu().detach().numpy())
        features_y.append(feature_y.cpu().detach().numpy()) 
    return features, features_y, modelo,new_obj
def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    if np_img.any():
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        return tensor

if __name__ == "__main__":
    new_y=[]
    face_ids=[]
    model_file = sys.argv[1]
    dataset_file = sys.argv[2]

    # Get the right files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    explan_dir = os.path.join(script_dir, 'grad_explanations')
    model_path = os.path.join(script_dir, 'models', model_file)
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)

    model_path = model_path
    image_path = dataset_path

    # Get embeddings and calculate cosine similarity
    features, features_y,modelo,obj=get_features(model_path,image_path,orig=False)
    similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
    pair_found=0
    sure_thing=0
    surepairs=0
    top_5=0
    i=0


    decisions=[]
    # Iterate over the probes, and find the maximum, while also checking if any similarity score is over a threshold for an accept
    for r in similarity_scores:
        decision=np.zeros(len(similarity_scores[0]))
        max_index=np.argmax(r)
        ind = np.argpartition(r, -5)[-5:]
        max_value=r[max_index]
        for sc in range(len(r)):
            score=r[sc]
            if score>0.4:
                decision[i]=1
                sure_thing+=1
                if i==sc:
                    surepairs+=1

        decisions.append(decision)
        if i in ind:
            top_5+=1
        if i==max_index:
            
            pair_found+=1
        #else:
            #print('mistake at: ', i)
        #print('face{} matches face {} the most (val={})'.format(i, max_index, max_value))
        i+=1
    wrong=sure_thing-surepairs
    print("rank_1 acc: ", round(pair_found/len(similarity_scores), 2))
    print("rank_5 acc: ",round(top_5/len(similarity_scores), 2))
    print("TAR: ", round(surepairs/len(similarity_scores), 2))
    print("FAR: ", round(wrong/len(similarity_scores), 2))

    print(surepairs)
    print(sure_thing)
