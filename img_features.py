import clip
import torch
from PIL import Image
import numpy as np
import pandas as pd


def get_image_features():
    '''
    Save the image features into a dataframe, which is further used as a reference to
    create the df including textual features.
    '''

    # GET ARRAY OF CLASSES
    classes = np.loadtxt("vocab.txt", dtype=str)

    # FLATTEN THE ARRAY TO A SINGLE ROW
    new_classes = []
    for i in range(len(classes)):
        for k in range(len(classes[0])):
            new_classes.append(classes[i][k])

    # LOAD CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    df = pd.read_json('labeled_instances.jsonl', lines=True)
    img_ft = np.zeros((1, len(new_classes)))


    for i in range(len(df)):
        if i%1000==0:
            print("Getting features of " + str(i) + "th Image")

        image = Image.open('data/'+df.loc[i]["image_number"])
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # PICK TOP 5 FEATURES FROM THE IMAGES
        text_features = torch.load('classes_features.pt')
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # GET THE RESULT
        temp_arr = np.zeros((1, len(new_classes)))
        for value, index in zip(values, indices):
            temp_arr[0][index] = value.item()
            # print(f"{new_classes[index]:>16s}: {100 * value.item():.2f}%")
        
        img_ft = np.vstack((img_ft, temp_arr))

    img_ft = np.delete(img_ft, 0, 0)

    image_df = pd.DataFrame(data=img_ft, columns=new_classes, index=np.arange(0, len(df)))
    result = pd.concat([df, image_df], axis=1)
    result.to_csv('image_data.csv')

# get_image_features()