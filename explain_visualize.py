import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gradient_calculator
from train import FaceResNet 
from lime import lime_image
import torch.nn as nn
import numpy as np
from eval import get_features
from skimage.segmentation import mark_boundaries
import sys
import os
#this is for the explanations
LIME_explainer = lime_image.LimeImageExplainer()

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
    for ind in range(len(ima)):
        im=Image.fromarray(ima[ind])
        x=transform(im).to(device)
        x=torch.unsqueeze(x, 0)
        _,feature = modelo.get_embed(x.float())
        features[ind]=feature.cpu().detach().numpy()
        new_similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
        # Return the similarity score for the specified index
        predicts.append(new_similarity_scores[ind])
    return torch.tensor(predicts)

def perturb_image(image, mask):
    perturbed_image = image.copy()
    perturbed_image[mask == 0] = 0  # Example perturbation by zeroing out unimportant regions
    return perturbed_image

def process_gradient(gradient):
    # Aggregate the gradient across the color channels
    # Option 1: Mean of absolute values
    aggregated_gradient = np.mean(np.abs(gradient), axis=2, keepdims=True)
    
    # Normalize the gradient
    aggregated_gradient -= aggregated_gradient.min()
    aggregated_gradient /= aggregated_gradient.max()

    return np.squeeze(aggregated_gradient)

def generate_mask(processed_gradient, threshold=0.5):
    # Apply threshold to create a binary mask
    mask = processed_gradient > threshold

    
    return mask



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <model_dir> <dataset_file>")
        sys.exit(1)

    model_dir = sys.argv[1]
    dataset_file = sys.argv[2]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    explan_dir = os.path.join(script_dir, 'explan')

    # Define paths relative to the script's directory
    names_path = os.path.join(script_dir, 'datasets', 'names.pkl')
    model_path = os.path.join(script_dir, 'models', model_dir)
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)
    low_qaulypeople,people = pd.read_pickle(names_path)
    features, features_y,modelo,obj=get_features(model_path,dataset_path,orig=False)
    similarity_scores = np.concatenate(features) @ np.concatenate(features_y).T
    im_obj = pd.read_pickle(dataset_path)

    indices=[]
    i=0
    for r in similarity_scores:
        max_index=np.argmax(r)
        indices.append(max_index)
        i+=1

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Explainer_model(modelo)
    g_truth=np.linspace(1, len(indices), num=len(indices))
    i=88

    im=obj[0][i]
    im_y=obj[1][indices[i]]

    ima=Image.fromarray(im_obj[0][i]).convert('RGB')
    tar_ima=Image.fromarray(im_obj[1][indices[i]]).convert('RGB')


    explanation = LIME_explainer.explain_instance(np.array(ima), classifier_fn, top_labels=1, hide_color=0, num_samples=500)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    plt.show()

        # Define paths relative to the script's directory
    combined_im_path = os.path.join(explan_dir, 'combined_im.png')
    original_im_path = os.path.join(explan_dir, 'original.png')
    target_im_path = os.path.join(explan_dir, 'target.png')
    new_im_path = os.path.join(explan_dir, 'new.png')

    pos=gradient_calculator.get_gradient(im.float(), im_y.float(), model, 1, 0.5)
    processed_gradient = process_gradient(pos)
    mask = generate_mask(processed_gradient, 0.5)

    neg=gradient_calculator.get_gradient(im.float(), im_y.float(), model, 2, 0.5)
    gradient_calculator.plot_gradient_2(pos,neg,explan_dir,0.5)
    overlay = Image.open(combined_im_path)
    background =ima
    background = background.convert("RGB")
    ima.save(original_im_path,"PNG")
    target_y=Image.fromarray(im_obj[1][i]).convert('RGB')
    target_y.save(target_im_path,"PNG")
    overlay = overlay.convert("RGB")
    new_img = Image.blend(background, overlay, 0.6)
    new_img.save(new_im_path,"PNG")




