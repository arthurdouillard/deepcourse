class Backbone(nn.Module):
  def __init__(self):
    super().__init__()

    self.convs = nn.Sequential(
        ConvBlock(3, 16, 3, 1),
        ConvBlock(16, 16, 3, 1),
        nn.MaxPool2d((2, 2)),
        ConvBlock(16, 32, 3, 1),
        ConvBlock(32, 32, 3, 1),
        nn.MaxPool2d((2, 2)),
        ConvBlock(32, 64, 3, 1),
        ConvBlock(64, 64, 3, 1),
        nn.MaxPool2d((2, 2)),
        ConvBlock(64, 128, 3, 1),
        ConvBlock(128, 128, 3, 1),
    )

    self.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 512),
        nn.Dropout(0.3),
        nn.Linear(512, 100)
    )

  def forward(self, x):
    x = self.convs(x)
    x = x.view(len(x), -1)
    return self.fc(x)
