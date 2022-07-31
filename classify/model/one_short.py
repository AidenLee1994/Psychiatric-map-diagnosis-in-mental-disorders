import torch
import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F


class SiameseNet(nn.Module):
    """
    A Convolutional Siamese Network for One-Shot Learning [1].
    Siamese networts learn image representations via a supervised metric-based
    approach. Once tuned, their learned features can be leveraged for one-shot
    learning without any retraining.
    References
    ----------
    - Koch et al., https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3)  #64 18 18
        self.conv2 = nn.Conv2d(64, 128, 3)  # 128 16 16
        self.conv3 = nn.Conv2d(128, 128, 3) #128 14 14
        self.conv4 = nn.Conv2d(128, 256, 3)  #256 12 12
        self.fc1 = nn.Linear(256 * 12 * 12, 4096)
        self.fc2 = nn.Linear(4096, 1)

        # self.conv1_bn = nn.BatchNorm2d(64)
        # self.conv2_bn = nn.BatchNorm2d(128)
        # self.conv3_bn = nn.BatchNorm2d(128)
        # self.conv4_bn = nn.BatchNorm2d(256)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')


    def sub_forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        out = sigmoid(self.fc1(out))
        return out

    def forward(self, x1, x2):

        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # compute l1 distance
        diff = torch.abs(h1 - h2)

        # score the similarity between the 2 encodings
        scores = self.fc2(diff)

        # return scores (without sigmoid) and use bce_with_logits
        # for increased numerical stability
        return scores