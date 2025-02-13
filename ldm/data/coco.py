import torch 
import pandas as pd
import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import csv
from PIL import Image
import requests
# !pip install StringIO
import io
from urllib.request import urlopen
from PIL import Image
import cv2
import os
import numpy as np


class CocoBase(Dataset):
    def __init__(self, 
                 path: str,
                size_of_dataset: int,
                width: int, 
                height: int):
        path = path.split('.')
        self.path = '/'.join(path)
        self.size_of_dataset = size_of_dataset
        self.width = width
        self.height = height
        counter = 0
        #TODO Add a cache parameter
        
        filepath = os.path.join(self.path,'Train-GCC-training.tsv')
        self.data = []
        
        with open(filepath) as file:
            tsv = csv.reader(file, delimiter='\t')
            for i,line in enumerate(tsv):
                caption, image_url = line[0], line[1]
                try:
                    file = urlopen(image_url)
                    arr = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, -1) 
                    #print(f'img {img.size()}')
                    img = cv2.resize(img, (self.width, self.height))
                    self.data.append({'caption': caption, 'image': img})
                    counter += 1
                except Exception as e:
                    if e:
                        print(f"EXCEPTION: {e}")
                    else:
                        print(f"EXCEPTION: message not found")
                    print(f'skipped image index {i}')
                if counter == self.size_of_dataset:
                    print(f'Dataset of number of elements {self.size_of_dataset}')
                    break
        return
  
    def __getitem__(self, 
                    i: int):
        return self.data[i]
    
    def __len__(self):
        return self.size_of_dataset
    
class CocoTrain(CocoBase):
    def __init__(self, path, size_of_dataset, width, height):
        super().__init__(path, size_of_dataset, width, height)

        
class CocoValidation(CocoBase):
    def __init__(self, path, size_of_dataset, width, height):
        super().__init__(path, size_of_dataset, width, height)

