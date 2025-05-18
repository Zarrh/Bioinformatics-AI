import torchvision.transforms as transforms
from constants import *

train_transform = transforms.Compose([
  transforms.Resize(target_size),
  transforms.RandomHorizontalFlip(),
  transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
  transforms.ColorJitter(brightness=(0.8, 1.2)),
  transforms.RandomRotation(10),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])