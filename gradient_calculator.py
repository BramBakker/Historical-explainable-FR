import numpy as np
import torch
import os

from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





def get_cosine(img_1, img_2, model, version, thr):
    """Compute cosine similarity of img_1 and img_2 with possibility to hide positive/negative features."""
    # get current embedding
    _,emb_1 = model.get_embed(img_1)
    _,emb_2 = model.get_embed(img_2)

    # normalize embedding
    emb_1 = nn.functional.normalize(emb_1, p=2.0, dim=1)
    emb_1 = emb_1.detach().cpu().numpy().squeeze()
    emb_2 = nn.functional.normalize(emb_2, p=2.0, dim=1)

    # remove irrelevant features
    if version == 0:  # standard cosine similarity computation
        emb_2 = emb_2.detach().squeeze()
        model.cosine_layer.weight = nn.Parameter(emb_2)
    elif version == 1:  # only keep positive features
        bound = thr / len(emb_1)
        emb_2 = emb_2.detach().cpu().numpy().squeeze()
        weights = torch.tensor([x[1] if x[0] >= bound else 0.0 for x in np.column_stack((np.multiply(emb_1, emb_2), emb_2))])
        weights = weights.to(device)
        model.cosine_layer.weight = nn.Parameter(weights)
    else:  # only keep negative features
        bound = thr / len(emb_1)
        emb_2 = emb_2.detach().cpu().numpy().squeeze()
        weights = torch.tensor([x[1] if x[0] < bound else 0.0 for x in np.column_stack((np.multiply(emb_1, emb_2), emb_2))])
        weights = weights.to(device)
        model.cosine_layer.weight = nn.Parameter(weights)

    img_rgb = torch.tensor(img_1, requires_grad=True, device=device)
    cos = model.cosine(img_rgb)
    return cos, img_rgb


def get_gradient(img1, img2, model, version, thr):
    """Compute gradient of img1 in model."""
    cos, root_1 = get_cosine(img1, img2, model, version, thr)

    feature = cos.squeeze()
    feature.backward(retain_graph=True)
    feature_gradients = root_1.grad
    fg = feature_gradients.detach().cpu().numpy().squeeze()

    fg = np.transpose(fg, (1, 2, 0))  # (height, width, channel)

    return fg

def load_gradient_mean(gradient):
    """
    Compute gradient map with RGB-channels combined via mean
    :param path_gradient: system path to gradient image
    :return: 2D matrix with RGB-channels combined with mean
    """

    return np.mean(np.abs(gradient), axis=2)

def plot_gradient_2(grad_pos_path, grad_neg_path, plot_path, masked):
    """
    Save positive, negative and combined gradient maps for image_path1
    :param img_path1: path to input image
    :param img_path2: path to reference image
    :param grad_pos_path: path to positive gradient argument file
    :param grad_neg_path: path to negative gradient argument file
    :param plot_path: save directory for gradient images
    :param masked: masked percentage
    :return:
    """
    id1 = "im_1"
    id2 = "im_2"

    # get mean gradient
    gradient_1_2_pos = load_gradient_mean(grad_pos_path)
    gradient_1_2_neg = load_gradient_mean(grad_neg_path)

    max_grad = max(map(max, [gradient_1_2_pos.flatten(), gradient_1_2_neg.flatten()]))
    min_grad = min(map(min, [gradient_1_2_pos.flatten(), gradient_1_2_neg.flatten()]))
    bins = 100
    # Define the size and DPI
    output_width, output_height = 112, 112 # Size in inches (since 1 inch = 100 pixels with 100 DPI)
    dpi = 100
    fig_size = (1.43, 1.425)
    print(fig_size)

#   Create a plot (example)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    ax.plot([0, 1], [0, 1])
    ax.axis('off')

    # visualize positive gradient map
    plt.imshow(gradient_1_2_pos, vmin=min_grad, vmax=max_grad, cmap='Greens')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'combined_im'),
                bbox_inches='tight',pad_inches=0, dpi=dpi)
    plt.close(fig)
    # visualize negative gradient maps
    plt.imshow(gradient_1_2_neg, vmin=min_grad, vmax=max_grad, cmap='Reds')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'neg_grad_input_{}_ref_{}_replace_{}pct'.format(id1, id2, int(masked * 100))),
                bbox_inches='tight')

    # visualize combined gradient maps
    combi = gradient_1_2_pos - gradient_1_2_neg
    bound = max(np.abs(min(map(min, combi))), max(map(max, combi)))
    plt.imshow(combi, vmin=bound * -1, vmax=bound, cmap='RdYlGn')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'combi_grad_input_{}_ref_{}_replace_{}pct'.format(id1, id2, int(masked * 100))),
                bbox_inches='tight')

    plt.close('all')