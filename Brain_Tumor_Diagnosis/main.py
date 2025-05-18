import os 
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models, datasets


from constants import *
from utils import *
from imageDataset import ImageDataset
from CNN import BrainTumorCNN
from transforms import *


# print(categories)
# display_images(train)


train_df = create_dataset(base_directory+'/'+train)
test_df = create_dataset(base_directory+'/'+test)

train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)


test_df_split, val_df_split = train_test_split(test_df, test_size=0.5, random_state=random_state)
# Reset indices for consistency
test_df_split = test_df_split.reset_index(drop=True)
val_df_split = val_df_split.reset_index(drop=True)


train_dataset = ImageDataset(train_df, transform=train_transform)
val_dataset = ImageDataset(val_df_split, transform=test_transform)
test_dataset = ImageDataset(test_df_split, transform=test_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = BrainTumorCNN(num_classes=4).to(device) 
model.load_state_dict(torch.load('model.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  


print("Starting training...")
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# η = 0.0007, n_epochs = 10 - 10 - 10

plot_training_history(history, model_name="CustomCNN")