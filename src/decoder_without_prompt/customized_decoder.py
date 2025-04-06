import torch.nn as nn

class SimpleMaskDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 输出为 1 通道（前景/背景）
        )

    def forward(self, x):
        return self.decoder(x)  # 输出尺寸为 (B, 1, H, W)