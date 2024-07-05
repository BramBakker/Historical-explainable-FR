import pandas as pd
from PIL import Image
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
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
    dataset_file = sys.argv[1]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'datasets', dataset_file)

    new_y=[]
    face_ids=[]
    image_path=  dataset_path
    obj = pd.read_pickle(image_path)
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

    sure_thing=0
    surepairs=0
    pair_found=0
    top_5=0
    i=0
    for r in similarity_scores:
        max_index=np.argmax(r)
        ind = np.argpartition(r, -5)[-5:]
        max_value=r[max_index]
        for sc in range(len(r)):
            score=r[sc]
            if score>0.4:
                sure_thing+=1
                if i==sc:
                    surepairs+=1
        if i in ind:
            top_5+=1
        if i==max_index:
            pair_found+=1
        print('face{} matches face {} the most (val={})'.format(i, max_index, max_value))
        i+=1
    wrong=sure_thing-surepairs
    print("rank_1 acc: ", round(pair_found/len(similarity_scores), 2))
    print("rank_5 acc: ",round(top_5/len(similarity_scores), 2))
    print("TAR: ", round(surepairs/len(similarity_scores), 2))
    print("FAR: ", round(wrong/len(similarity_scores), 2))

    print(surepairs)
    print(sure_thing)

