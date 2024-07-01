# Historical-explainable-FR
This codebase is used for experiments of various FR loss functions and data augmentation methods in the domain of historical data, both head.py and net.py are adopted from Adaface https://github.com/mk-minchul/AdaFace, while gradient_calculator.py originates from https://github.com/marcohuber/xSSAB.

# Data availability
The dataset can be shared upon request

# How to use
To train the model under various conditions, change out the head of the model in the code. The train file is a tuple of two lists, one of images and one of target classes.<br />
run the following command: train.py  <path_tosavemodel> <train_file> <br />


To train the GAN, use the same train file <br />
run: train_GAN.py <path_to_save_model> <train_file>

To train with triplet loss, a custom train file of a single list of triplets is needed<br /> 
run: traintrip.py <path_to_save_model> <train_file>

To generate saliency maps from a particular model where the validation_file consists of a tuple, a list of probe images, and a gallery of target images<br /> 
run: explanation.py <path_to_model> <validation_file> <br /> 
. 

To calculate the DAUC and IAUC of the generated saliency maps<br /> 
run: eval_explanation.py  <path_to_model> <explanation_files>

To visualize a particular saliency map<br /> 
run: explain_visualize.py <path_to_model> <validation_file> 

To evaluate a particular model<br /> 
run: eval.py <path_to_model> <validation_file> 

