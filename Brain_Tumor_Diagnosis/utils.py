import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

import torch

from constants import *


def display_images(dataset_type, num_images=4, image_size=(224, 224)):
    
  dataset_path = os.path.join(base_directory, dataset_type)

  fig, axes = plt.subplots(len(categories), num_images, figsize=(15, 10))

  for row, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    image_filenames = random.sample(os.listdir(category_path), num_images)  # Select random images
    
    for col, image_filename in enumerate(image_filenames):
      while image_filename == '.DS_Store':
        image_filename = random.sample(os.listdir(category_path), 1)[0]
      image_path = os.path.join(category_path, image_filename)
      image = Image.open(image_path).resize(image_size)
      axes[row, col].imshow(image, cmap='gray')
      axes[row, col].axis('off')
      axes[row, col].set_title(f"{category}")

  plt.tight_layout()
  plt.show()


def plot_class_distribution(dataset_type):
  path = os.path.join(base_directory, dataset_type)
  counts = [len(os.listdir(os.path.join(path, cat))) for cat in categories]

  plt.bar(categories, counts, color = ['navy', 'teal', 'darkorange', 'crimson'])
  plt.xlabel("Class")
  plt.ylabel("Number of Images")
  plt.title(f"{dataset_type.capitalize()} Set Distribution")
  plt.show()


def create_dataset(path):
  my_list = []
  for category in categories:
    category_path = os.path.join(path, category)
    for file_name in os.listdir(category_path):
      file_path = os.path.join(category_path, file_name)
      if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        my_list.append([file_path, category])
  return pd.DataFrame(my_list, columns=['file_path', 'label'])



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, name='model', patience=7):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  best_val_loss = float("inf")
  tolerance = 0 
  history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

  for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total_train += labels.size(0)
      correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("#" * 80)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), f'best_brain_tumor_{name}.pth')
      tolerance = 0  
    else:
      tolerance += 1
      if tolerance >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

  return history



def test_model(model, test_loader, num_images_to_show=10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  
  correct = 0
  total = 0
  
  all_preds = []
  all_labels = []
  all_images = []

  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      all_images.extend(images.cpu())

  test_acc = 100 * correct / total

  cm = confusion_matrix(all_labels, all_preds)

  print(f"Test Accuracy: {test_acc:.2f}%\n")
  
  print("Classification Report:\n")
  print(classification_report(all_labels, all_preds, target_names=categories))

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
      xticklabels=categories, yticklabels=categories, annot_kws={"color": "grey", "weight": "bold"})
  plt.title('Confusion Matrix', color='grey', weight='bold')
  plt.xlabel('Predicted Label', color='grey', weight='bold')
  plt.ylabel('True Label', color='grey', weight='bold')
  plt.xticks(color='grey', fontweight='bold')
  plt.yticks(color='grey', fontweight='bold')
  plt.tight_layout()
  plt.savefig("confusion_matrix.png", transparent=True)
  plt.close()
  plt.show()


def plot_training_history(history, model_name="model"):
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(history['train_loss'], label='Train Loss')
  plt.plot(history['val_loss'], label='Val Loss')
  plt.title(f'{model_name} Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(history['train_acc'], label='Train Acc')
  plt.plot(history['val_acc'], label='Val Acc')
  plt.title(f'{model_name} Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()
  plt.show()