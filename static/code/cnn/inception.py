class InceptionBlock(nn.Module):
  def __init__(self, in_channels, reduced_channels, out_channels):
    super().__init__()

    self.branch1x1 = ConvBlock(in_channels, out_channels, 1)

    self.branch3x3_1 = ConvBlock(in_channels, reduced_channels, 1)
    self.branch3x3_2 = ConvBlock(reduced_channels, out_channels, 3, padding=1)

    self.branch5x5_1 = ConvBlock(in_channels, reduced_channels, 1)
    self.branch5x5_2 = ConvBlock(reduced_channels, out_channels, 5, padding=2)

    self.branch_pool = ConvBlock(in_channels, out_channels, 1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch3x3 = self.branch3x3_1(x)
    branch3x3 = self.branch3x3_2(branch3x3)

    branch5x5 = self.branch5x5_1(x)
    branch5x5 = self.branch5x5_2(branch5x5)

    branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    return torch.cat(
        (branch1x1, branch3x3, branch5x5, branch_pool),
        dim=1
    )
