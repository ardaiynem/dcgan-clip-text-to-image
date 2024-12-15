import torch
import torch.nn as nn
from config import Config

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, embedding_dim):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.embedding_dim = embedding_dim
        self.image_net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4 + embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, embeddings):
        x = self.image_net(image)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = torch.cat((x, embeddings), 1)
        output = self.fc(x)
        return output.squeeze()