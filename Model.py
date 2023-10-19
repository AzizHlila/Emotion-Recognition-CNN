import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import PIL.Image 

class CNN(nn.Module):
    
    # Contructor
    def __init__(self, out_1=16, out_2=32):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4 * 9, 7)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



class Model:
    PATH = "Models/model.AI"
    df_features = pd.read_csv("testDataFeatures.csv")
    IMAGE_SIZE = 48
    compose = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor()])

    def getMostEmotion(self,yhat):
        self.df_features["val"] = yhat.detach().numpy()[0]
        return self.df_features["0"][np.where(self.df_features["val"].max()==self.df_features["val"])[0]] .to_string().split(":")[-1]

    def __init__(self) -> None:
        self.model = CNN(out_1=16, out_2=32)
        self.model.load_state_dict(torch.load(self.PATH))
        #self.model.eval()
    
    def prepareImage(self,img):
        '''
        X : PIL image
        '''
        img2 = self.compose(img)

        if(img2.shape[0]>1):
            img2 = img2[0:1]

        return img2.view((1,1,48,48))


    def predict(self,img):
        '''
        X : PIL image
        '''
        img2 = self.prepareImage(img)
        z = self.model(img2)
        # _, yhat = z.max(1)
        return self.getMostEmotion(z),z
    
