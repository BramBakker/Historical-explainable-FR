import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gradient_calculator
from lime import lime_image
import torch.nn as nn
import numpy as np
from eval import get_features
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from google import protobuf
from skimage.segmentation import slic
import pickle
import os
import sys
def get_superpixel_weights(image, clas_fn):
    explanation = LIME_explainer.explain_instance(image, clas_fn, top_labels=1, hide_color=0, num_samples=300)
    top_label = explanation.top_labels[0]
    weights = dict(explanation.local_exp[top_label])
    return weights

def create_weight_mask(image, weights):
    segments = slic(image, n_segments=100, compactness=10)
    weight_mask = np.zeros(segments.shape)
    for superpixel, weight in weights.items():
        weight_mask[segments == superpixel] = weight
    return weight_mask

def classifier_fn(ima):
    predicts=[]
    for image in ima:
        im=Image.fromarray(image).convert('RGB')
        feature = DeepFace.represent(np.array(im), enforce_detection=False,model_name=models[8])
        index=i
        features[index]=feature[0]["embedding"]
        if np.isnan(features[index]).any():
            continue
        else:
            new_similarity_scores = cosine_similarity(features,features_y)
            # Return the similarity score for the specified index
            predicts.append(new_similarity_scores[index])
    return torch.tensor(predicts)



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
    
    #this is for the explanations
    LIME_explainer = lime_image.LimeImageExplainer()
    model_file = "LIME_expl_SFace.pkl"
    dataset_file = "valface.pkl"

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the script's directory
    model_path = os.path.join(script_dir, 'models', model_file)
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)

    new_y=[]
    face_ids=[]
    obj = pd.read_pickle(dataset_path)
    features=[]
    features_y=[]
    for i in range(len(obj[0])):
        im=Image.fromarray(obj[0][i]).convert('RGB')
        im_y=Image.fromarray(obj[1][i]).convert('RGB')
        feature = DeepFace.represent(np.array(im), enforce_detection=False,model_name=models[8])
        feature_y = DeepFace.represent(np.array(im_y), enforce_detection=False,model_name=models[8])
        features.append(feature[0]["embedding"])
        features_y.append(feature_y[0]["embedding"])
    similarity_scores=cosine_similarity(features,features_y)

    indices=[]

    for r in similarity_scores:
        max_index=np.argmax(r)
        indices.append(max_index)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g_truth=np.linspace(1, len(indices), num=len(indices))
    explanations_LIME=[]
    for i in range(len(indices)):

        ima=Image.fromarray(obj[0][i]).convert('RGB')

        weights = get_superpixel_weights(np.array(ima),classifier_fn)
        weight_masks = create_weight_mask(np.array(ima),weights)

        explanations_LIME.append(weight_masks)
with open(model_path, 'wb') as fp:
    pickle.dump(explanations_LIME, fp)