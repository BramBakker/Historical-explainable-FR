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
from scipy.special import softmax
from train import load_model
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import sys, getopt
import pickle
from skimage.segmentation import slic
import os
import sys


class Explainer_model(nn.Module):
    def __init__(self, original_model):
        super(Explainer_model, self).__init__()
        self.input_layer = original_model.input_layer
        self.body = original_model.body
        self.output_layer = original_model.output_layer
        self.cosine_layer = nn.Linear(512, 1, bias=False)

    def get_embed(self, x):
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norms = torch.norm(x, 2, 1, True)
        features = torch.div(x, norms)
        return norms,features
    
    def cosine(self,x):
        _,x = self.get_embed(x)
        x = nn.functional.normalize(x, p=2.0, dim=1)
        x = self.cosine_layer(x)

        return x


def classifier_fn(ima):
    predicts=[]
    orig=False
    for ind in range(len(ima)):
        im=Image.fromarray(ima[ind])
        x=transform(im).to(device)
        x=torch.unsqueeze(x, 0)
        if orig==True:
            _,feature = modelo.get_embed(to_input(ima[ind])[0].to(device).float())
        else:
            _,feature = modelo.get_embed(x.float())
        features[ind]=feature.cpu().detach().numpy()
        new_similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
        # Return the similarity score for the specified index
        predicts.append(new_similarity_scores[ind])
    return torch.tensor(predicts)










def process_gradient(gradient):
    # Aggregate the gradient across the color channels
    # Option 1: Mean of absolute values
    aggregated_gradient = np.mean(np.abs(gradient), axis=2, keepdims=True)
    
    # Normalize the gradient
    aggregated_gradient -= aggregated_gradient.min()
    aggregated_gradient /= aggregated_gradient.max()

    return np.squeeze(aggregated_gradient)

def generate_mask(processed_gradient, threshold=0.5):
    mask = processed_gradient > threshold
    return mask
def get_superpixel_weights(image, clas_fn):
    explanation = LIME_explainer.explain_instance(image, clas_fn, top_labels=1, hide_color=0, num_samples=200)
    top_label = explanation.top_labels[0]
    weights = dict(explanation.local_exp[top_label])
    return weights

def create_weight_mask(image, weights):
    segments = slic(image, n_segments=100, compactness=10)
    weight_mask = np.zeros(segments.shape)
    for superpixel, weight in weights.items():
        weight_mask[segments == superpixel] = weight
    return weight_mask
if __name__ == "__main__":
    model_file = sys.argv[1]
    dataset_file = sys.argv[2]
    # Get the right files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    explan_dir = os.path.join(script_dir, 'grad_explanations')

    names_path = os.path.join(script_dir, 'datasets', 'names.pkl')
    model_path = os.path.join(script_dir, 'models', model_file)
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    expl_LIME_path= os.path.join(explan_dir, model_file+"_LIME_expl.pkl")
    expl_xssab_path= os.path.join(explan_dir, model_file+"_xssab_expl.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate embeddings
    features, features_y,modelo,obj=get_features(model_path,dataset_path, orig=False)
    similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
    im_obj = pd.read_pickle(dataset_path)

    indices=[]
    # Define the LIME explainer
    LIME_explainer = lime_image.LimeImageExplainer()

    for r in similarity_scores:
        max_index=np.argmax(r)
        indices.append(max_index)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Explainer_model(modelo)
    explanations_LIME=[]
    explanations_xxsab=[]
    # For each image get a prediction from the max cosine similarity and calculate saliency maps
    for i in range(len(indices)):
        im=obj[0][i]
        im_y=obj[1][indices[i]]

        ima=Image.fromarray(im_obj[0][i]).convert('RGB')
        weights = get_superpixel_weights(np.array(ima),classifier_fn)
        weight_masks = create_weight_mask(np.array(ima),weights)
        positive_weights = {superpixel: weight for superpixel, weight in weights.items() if weight > 0}

        explanations_LIME.append(positive_weights)
        pos=gradient_calculator.get_gradient(im.float(), im_y.float(), model, 1, 0.5)
        processed_gradient = process_gradient(pos)
        mask = generate_mask(processed_gradient, 0.5)
        explanations_xxsab.append(processed_gradient)
        print(i)

    # Save the saliency maps for use in eval_explanation.py
    with open(expl_LIME_path, 'wb') as fp:
        pickle.dump(explanations_LIME, fp)

    with open(expl_xssab_path, 'wb') as fp:
        pickle.dump(explanations_xxsab, fp)
