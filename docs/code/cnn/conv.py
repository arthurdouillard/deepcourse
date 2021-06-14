class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=padding)
    # We don't need the bias when used with BN because the BN also contains a bias
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=True)
