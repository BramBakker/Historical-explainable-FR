import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gradient_calculator
from train import FaceResNet 
from lime import lime_image
import torch.nn as nn
import numpy as np
from eval import get_features
import pickle
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# With this we use the predict function for other SOTA models
def classifier_others(ima):
    predicts=[]
    i=0
    for image in ima:
        im=Image.fromarray(image).convert('RGB')
        feature = DeepFace.represent(np.array(im), enforce_detection=False,model_name=models[8])
        index=i
        i+=1
        features[index]=feature[0]["embedding"]
        if np.isnan(features[index]).any():
            continue
        else:
            new_similarity_scores = cosine_similarity(features,features_y)
            # Return the similarity score for the specified index
            predicts.append(new_similarity_scores[index])
    return torch.tensor(predicts)

# The predicts are just the cosine similarity scores
def classifier_fn(ima,orig):
    predicts=[]
    
    for ind in range(len(ima)):
        im=Image.fromarray(ima[ind])
        x=transform(im).to(device)
        x=torch.unsqueeze(x, 0)
        if orig==True:
            _,feature = modelo.get_embed(to_input(im)[0].to(device).float())
        else:
            _,feature = modelo.get_embed(x.float())
        features[ind]=feature.cpu().detach().numpy()
        new_similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
        # Return the similarity score for the specified index
        predicts.append(new_similarity_scores[ind])
    return predicts
def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    if np_img.any():
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        return [tensor, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
HW = 112 * 112

# This is how we blur the image
def gkern(klen, nsig):
    inp = np.zeros((klen, klen))
    inp[klen//2, klen//2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

klen = 11
ksig = 5
kern = gkern(klen, ksig)
transform = transforms.Compose([
    transforms.PILToTensor()
])
def blur(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()  # Convert to PyTorch tensor
    if x.ndim == 3:  # Add batch dimension if missing
        x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
    blurred =nn.functional.conv2d(x, kern, padding=klen // 2).permute(0, 2, 3, 1)
    return blurred.to(torch.uint8)# Change

def auc(arr):
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


# This class is used to calculate either DAUC or IAUC
class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def evaluate(self, predictions, img_batch, exp_batch):
        n_samples = img_batch.shape[0]
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = self.substrate_fn(img_batch).cpu().numpy().copy()

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.copy()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.copy()
        print('finish',finish.shape)
        print('start',start.shape)
        finish = np.array(finish)
        for i in tqdm(range(n_steps + 1), desc=caption + 'pixels'):
            preds=classifier_fn(start,orig=False)
            for j in range(len(predictions)):
                probs= preds[j]
                scores[i][j] = probs[predictions[j]]

            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.reshape(n_samples, 3, HW)[r, :, coords] = finish.reshape(n_samples, 3, HW)[r, :, coords]
        return scores
if __name__ == "__main__":

    models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    "GhostFaceNet",
    ]

    model_file = sys.argv[1]
    dataset_file = sys.argv[2]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    explan_dir = os.path.join(script_dir, 'grad_explanations')

    # Define paths relative to the script's directory
    model_path = os.path.join(script_dir, 'models', model_file)
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    expl_LIME_path= os.path.join(explan_dir, model_file+"_LIME_expl.pkl")
    expl_xssab_path= os.path.join(explan_dir, model_file+"_xssab_expl.pkl")



    with open(expl_LIME_path, "rb") as fp:
        LIME_expl = pickle.load(fp)
    fp.close()
    with open(expl_xssab_path, "rb") as fp:
        xssab_expl = pickle.load(fp)
    fp.close()
    indices=[]
    print(np.array(LIME_expl[1]).shape)

    
    # Get similarity scores
    features, features_y,modelo,obj=get_features(model_path,dataset_path,orig=False)
    similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
    im_obj = pd.read_pickle(dataset_path)


    # This code is to get the similarity scores from the other SOTA models
    #image_path =  dataset_path
    #obj = pd.read_pickle(image_path)
    #features=[]
    #features_y=[]
    #for i in range(len(obj[0])):
        #im = Image.fromarray(obj[0][i]).convert('RGB')
        #im_y = Image.fromarray(obj[1][i]).convert('RGB')
        #feature = DeepFace.represent(np.array(im), enforce_detection=False,model_name=models[8])
        #feature_y = DeepFace.represent(np.array(im_y), enforce_detection=False,model_name=models[8])
        #features.append(feature[0]["embedding"])
    # features_y.append(feature_y[0]["embedding"])
    #similarity_scores=cosine_similarity(features,features_y)

    indices=[]

    for r in similarity_scores:
        max_index=np.argmax(r)
        indices.append(max_index)
    step = 150  # Number of pixels to perturb at each step
    substrate_fn = blur 


    # Calculate the DAUC and IAUC for both LIME and xSSAB
    causal_metric = CausalMetric(modelo, 'del', step, substrate_fn)
    causal_metric_ins = CausalMetric(modelo, 'ins', step, substrate_fn)

    
    L_ins_scores = causal_metric_ins.evaluate(indices, np.array(im_obj[0]), np.array(LIME_expl))
    mean_scores = L_ins_scores.mean(axis=1)
    print(mean_scores)
    single_value_auc = auc(mean_scores)
    print('L_ins_score: ', single_value_auc)
    L_del_scores = causal_metric.evaluate(indices, np.array(im_obj[0]), np.array(LIME_expl))
    mean_scores = L_del_scores.mean(axis=1)
    single_value_auc = auc(mean_scores)
    print(mean_scores)
    print('L_del_score: ', single_value_auc)

    L_ins_scores = causal_metric_ins.evaluate(indices, np.array(im_obj[0]), np.array(xssab_expl))
    mean_scores = L_ins_scores.mean(axis=1)
    print(mean_scores)
    single_value_auc = auc(mean_scores)
    print('x_ins_score: ', single_value_auc)
    L_del_scores = causal_metric.evaluate(indices, np.array(im_obj[0]), np.array(xssab_expl))
    mean_scores = L_del_scores.mean(axis=1)
    single_value_auc = auc(mean_scores)
    print(mean_scores)
    print('x_del_score: ', single_value_auc)



