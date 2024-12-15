import torch
import torch.nn as nn
from config import Config

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, embedding_dim):
        super(Generator, self).__init__()
        self.nz = nz
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(nz + embedding_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),       # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),            # 64x64
            nn.Tanh(),
        )

    def forward(self, noise, embeddings):
        x = torch.cat((noise, embeddings), 1)
        x = self.fc(x)
        x = x.view(-1, Config.ngf * 8, 4, 4)
        output = self.main(x)
        return output