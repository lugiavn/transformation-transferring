
import torch
import torchvision
import numpy as np
import random
import os
import sys
sys.path.append(os.getcwd() + "/tirg")
import text_model
import torch_functions

class MTirgTransform(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MTirgTransform, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim * 5),
            torch.nn.BatchNorm1d(embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim)
        )
        self.norm = torch_functions.NormalizationLayer(learn_scale=False)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 0.1]))

    def forward(self, x):
        #x = [x[0], x[0], x[2]]
        f = torch.cat([self.norm(i) for i in x], dim=1)
        f = self.m(f)
        f = self.a[0] * self.norm(x[0]) + self.a[1] * self.norm(f)
        return f

class ConcatTransform(torch.nn.Module):
    def __init__(self, embed_dim):
        super(ConcatTransform, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim * 5),
            torch.nn.BatchNorm1d(embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim)
        )
        self.norm = torch_functions.NormalizationLayer(learn_scale=False)

    def forward(self, x):
        f = torch.cat([self.norm(i) for i in x], dim=1)
        f = self.m(f)
        return f

class TirgTransform(torch.nn.Module):
    def __init__(self, embed_dim):
        super(TirgTransform, self).__init__()
        self.m1 = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim * 5),
            torch.nn.BatchNorm1d(embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim)
        )
        self.m2 = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 3, embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim * 5),
            torch.nn.BatchNorm1d(embed_dim * 5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 5, embed_dim)
        )
        self.norm = torch_functions.NormalizationLayer(learn_scale=False)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 0.05]))

    def forward(self, x):
        f = torch.cat([self.norm(i) for i in x], dim=1)
        f1 = self.m1(f)
        f2 = self.m2(f)
        f2 = torch.sigmoid(f2 / 10.0)
        f = self.a[0] * self.norm(x[0] * f2) + self.a[1] * self.norm(f1)
        return f

class ImageTextEncodeTransformModel(torch.nn.Module):
    
    def __init__(self, embed_dim, texts):
        super(ImageTextEncodeTransformModel, self).__init__()

        self.snorm = torch_functions.NormalizationLayer(normalize_scale=4.0, learn_scale=True)

        # image
        self.img_encoder = torchvision.models.resnet50(pretrained=True)
        self.img_encoder.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, embed_dim)
        )

        # text
        self.text_encoder = text_model.TextLSTMModel(
            texts_to_build_vocab = texts,
            word_embed_dim = 256,
            lstm_hidden_dim = embed_dim
        )
        self.text_encoder.fc_output = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, embed_dim)
        )

        # transformer
        self.transformer = MTirgTransform(embed_dim)


    def pair_loss(self, a, b):
        # force a,b similar in the embedding space
        a = self.snorm(a)
        b = self.snorm(b).transpose(0,1)
        x = torch.mm(a, b)
        if random.random() > 0.5:
            x = x.transpose(0, 1)
        labels = torch.tensor(range(x.shape[0])).long()
        return torch.nn.functional.cross_entropy(x, labels.cuda())
        



