# Validation Set
!wget http://images.cocodataset.org/zips/val2014.zip
!unzip val2014.zip

"""Captions """

import json

train, val, test = {}, {},{}   # Dicionaries for storing training and test data
all_captions = []
# Opening JSON file
with open('dataset_coco.json') as json_file:
    data = json.load(json_file)
     
    # for printing the key-value pair of nested dictionary for loop can be used
    for i in data['images']:
        
        if i['split'] == 'val':
          val[i['filepath']+"/"+i['filename']] = i
        elif i['split'] == 'test':
          test[i['filepath']+"/"+i['filename']] = i
          
          all_captions.append(i['sentences'][0]['raw'])
          all_captions.append(i['sentences'][1]['raw'])
          all_captions.append(i['sentences'][2]['raw'])
          all_captions.append(i['sentences'][3]['raw'])
          all_captions.append(i['sentences'][4]['raw'])

        else:
          train[i['filepath']+"/"+i['filename']] = i

print(len(val), len(test), len(train)) 

# Output Test

with open("test_coco.json", "w") as outfile_test:
    json.dump(test, outfile_test)
    

"""# **CLIP**"""

!pip3 install clip-by-openai



import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model and image preprocessing
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

import json
import numpy as np
# label_probs = []
label_probs = np.zeros((5, 25000))
# Opening JSON file
with open('test_coco.json') as json_file:
    data = json.load(json_file)
    ### loop here
    for i in range(5):
      text = clip.tokenize(all_captions[5000*i:5000*(i+1)]).to(device)  

      # for printing the key-value pair of nested dictionary for loop can be used
      image_count = 0
      for path,v in data.items():
        print(f'Image count is {image_count} and i is {i}')
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device) 
        with torch.no_grad():
          image_features = model.encode_image(image)
        with torch.no_grad():
          text_features = model.encode_text(text) 
        with torch.no_grad():
          logits_per_image, logits_per_text = model(image, text)
          probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # label_probs.append(probs)
        label_probs[image_count][5000*i:5000*(i+1)] = np.array(probs)
        image_count += 1
# print("Label probs:", label_probs)

from numpy import save

save('probs_data.npy', label_probs)
print("DONE, np array saved")