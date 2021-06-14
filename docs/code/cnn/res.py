class ResBlock(nn.Module):
  def __init__(self, input_channels, hidden_channels):
    super().__init__()
    # Use a kernel size of 3

    self.conv1 = ConvBlock(input_channels, hidden_channels, kernel_size=3, padding=1)
    self.conv2 = ConvBlock(hidden_channels, input_channels, kernel_size=3, padding=1)

  def forward(self, x):
    h = self.conv1(x)
    h = self.conv2(h)

    return x + h
