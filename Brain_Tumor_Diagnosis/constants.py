import torch
import os

base_directory = './data/'
train, test = 'Training', 'Testing'
target_size = (224, 224)
random_state = 42
batch_size = 32
num_classes = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
label_map = {
  'notumor': 0,        
  'glioma': 1,         
  'meningioma': 2,     
  'pituitary': 3       
}
categories = os.listdir(base_directory+'/'+train)