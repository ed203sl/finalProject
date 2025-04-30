import torch
import torch.nn as nn
import torch.nn.Functional as F
import random

# WARNING: CHANGING THIS SEED WILL RESULT IN A LOSS OF REPRODUCIBILITY
RANDOM_SEED = 20250501234
random.seed(RANDOM_SEED)
torch.manual_seed(202505)



class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='crossdomain'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(len(perm_b))]
            perm_a = perm_a[torch.randperm(len(perm_a))]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix





class ResBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=False):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout1d(0.2) if dropout else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
    
class ResNet1D(nn.Module):
    def __init__(self, block, layers, in_channels=12, num_classes=20, dropout=True, use_mixstyle=False):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.use_mixstyle = use_mixstyle
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        if dropout:
            self.layer1 = self._make_layer(block, 64, layers[0], dropout=True)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=True)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=True)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], dropout=False)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=False)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=False)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        if dropout:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(512*block.expansion, num_classes),
            )
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout=False):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout=dropout))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        if not(self.dropout):
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

