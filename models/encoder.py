import torch.nn as nn
from models.layers import ResNet18

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = ResNet18(pretrained=True)  # 64 -> 512
    
    def forward(self, images):
        '''
            x: [B, N, H, W, 3]
        '''
        B, N = images.shape[:2]
        x = images.flatten(0, 1).permute(0, 3, 1, 2)
        features = self.conv_blocks(x)
        _, C, H, W = features.shape
        return features.reshape(B, N, C, H, W)
