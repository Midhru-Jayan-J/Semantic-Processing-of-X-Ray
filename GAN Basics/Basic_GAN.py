import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
 
 
class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc =nn.Sequential(                           
            nn.Linear(img_dim,128),
            nn.LeakyReLU(img_dim,128),
            nn.Linear(128,1),
            nn.Sigmoid(),         
        )
        
        def forward(self,x):
            return self.disc(x)
        
        
        