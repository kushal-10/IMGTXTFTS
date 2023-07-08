import pandas as pd
import numpy as np
import clip
import torch
import numpy as np
from torch import concat

def save_classes():
    '''
    Takes the list of most frequent words in the captions (after removing stop words) 
    and generates a vocab.txt file with top K most occurring classes

    '''

    # CREATE A LIST OF TOP 1000 MOST OCCURING CLASSES
    top_k = 1000
    classes_df = pd.read_csv('token_frequencies.txt', sep='\t', header=None, names=["Word", "Score"], encoding='latin-1')
    frequent_vocab = []
    for i in range(top_k):
        frequent_vocab.append(classes_df.loc[i]["Word"])

    # CREATE BATCHES OF SIZE 50 / 1050Ti GPU LIMIT TO CREATE TENSORS
    new_classes = []
    temp_list = []
    i = 1
    for cl in frequent_vocab:
        temp_list.append(cl)
        if i%50==0:
            new_classes.append(temp_list)
            temp_list = []
        i += 1
    
    # SAVE THE GENERATED 2-D ARRAY
    new_classes = np.array(new_classes)
    np.savetxt('vocab.txt', new_classes, fmt="%s")

def save_classes_tensor():
    '''
    Takes the input of top K most occuring words, uses them as classes,
    and generates a tensor  
    '''

    # LOAD THE 2-D ARRAY CONTAINING CLASSES
    classes = np.loadtxt("vocab.txt", dtype=str)

    # MAKE A FLATTENED COPY OF THE 2-D ARRRAY, FOR REFERENCING INDEX
    new_classes = []
    for i in range(len(classes)):
        for k in range(len(classes[0])):
            new_classes.append(classes[i][k])

    # print(np.shape(classes), np.shape(new_classes))

    # LOAD CLIP AND CREATE CLASS FEATURES WHICH ARE USED AS INPUTS TO EXTRACT VISUAL FEATURES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # GENERATE TENSORS IN BATCHES OF 50
    # FLATTEN AT THE END FOR REFERENCING
    classes_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes[0]]).to(device)
    with torch.no_grad():
        classes_features = model.encode_text(classes_inputs)

    for i in range(1, len(classes)):
        classes_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes[i]]).to(device)
        # # Calculate features
        with torch.no_grad():
            t_features = model.encode_text(classes_inputs)
        classes_features = concat((classes_features, t_features), dim=0)

    # SAVE THE FLATTENED TENSOR
    torch.save(classes_features, 'classes_features.pt')

# save_classes()
# save_classes_tensor()







