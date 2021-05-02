import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self,img_root,label_root):
        super(dataset,self).__init__()
        self.img_root=img_root
        self.label_root=label_root
        self.img_paths=os.listdir(img_root)
        
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self,idx):
        img=Image.open(os.path.join(self.img_root,self.img_paths[idx]))
        label=np.array(Image.open(os.path.join(self.label_root,self.img_paths[idx])))
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        img=transform(img)
        return np.array(img),torch.from_numpy(label).long()

def test():
    dirname = os.path.dirname(__file__)
    train_img_root=os.path.abspath(os.path.join(dirname,"../CamVid/train/"))
    train_label_root=os.path.abspath(os.path.join(dirname,"../CamVid/trainannot/"))
    train_ds=dataset(train_img_root,train_label_root)
    print(train_ds[0][0].shape)
    print(train_ds[0][1].shape)
    print("HELLO")
    plt.imshow(train_ds[0][0])
    plt.show()
   
if __name__=='__main__':
    test()
