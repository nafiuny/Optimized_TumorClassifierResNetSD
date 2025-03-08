import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from sklearn.metrics import precision_score, recall_score


class TumorClassifierResNetSD(nn.Module):
    def __init__(self, num_classes, stochastic_depth1=0.6, stochastic_depth2=0.7):
        super(TumorClassifierResNetSD, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.resnet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            StochasticDepth(stochastic_depth1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            StochasticDepth(stochastic_depth2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.fc(features)
        return output



class StochasticDepth(nn.Module):
    def __init__(self, survival_prob):
        """
        Stochastic Depth Layer
        Args:
            survival_prob (float): 
        """
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if not self.training or torch.rand(1).item() < self.survival_prob:
            return x
        else:
            return torch.zeros_like(x)



