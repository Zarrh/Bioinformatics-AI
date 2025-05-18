from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
  def __init__(self, dataframe, transform=None):
    self.dataframe = dataframe
    self.transform = transform
      
  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    img_path = self.dataframe.iloc[idx, 0]
    label = self.dataframe.iloc[idx, 1]
    img = Image.open(img_path).convert('RGB')  

    if self.transform:
      img = self.transform(img)
        
    return img, label